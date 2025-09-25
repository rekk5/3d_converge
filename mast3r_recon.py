import os
import torch
import math
import glob
from pathlib import Path
import numpy as np
import trimesh
import copy
import shutil
import tempfile
import sys
from scipy.spatial.transform import Rotation
from PIL import Image

# Add MAST3R directory to Python path
MAST3R_ROOT = "/home/t/mast3r"
if MAST3R_ROOT not in sys.path:
    sys.path.insert(0, MAST3R_ROOT)

# Import MAST3R dependencies
try:
    from dust3r.dust3r.utils.image import load_images
    from dust3r.dust3r.image_pairs import make_pairs
    from dust3r.dust3r.utils.device import to_numpy
    from dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    from mast3r.model import AsymmetricMASt3R
    from mast3r.utils.misc import hash_md5
    MAST3R_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MAST3R dependencies not available: {e}")
    print("MAST3R reconstruction will not be available")
    print(f"Make sure MAST3R is installed at: {MAST3R_ROOT}")
    MAST3R_AVAILABLE = False

# For GPU >= Ampere and pytorch >= 1.12
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

class FileState:
    def __init__(self, outfile_name=None):
        self.outfile_name = outfile_name

    def __del__(self):
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    print("Number of cameras:", len(cams2world))
    print("Number of geometries in scene:", len(scene.geometry))
    print("First few points (if pointcloud):", pts[:5] if as_pointcloud else "N/A")
    print("First few colors (if pointcloud):", col[:5] if as_pointcloud else "N/A")
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        print("Scene_state None")
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        print("Outfile None")
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    print("Shape of pts3d[0]:", pts3d[0].shape)
    print("Mean confs[0]:", np.mean(confs[0]))
    print("Mask coverage ratio (first image):", np.mean(confs[0] > min_conf_thr))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
                            filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    return scene_state, outfile

def get_image_files_from_folder(folder_path, supported_extensions=None):
    """
    Get all image files from a folder, sorted by filename.
    
    Args:
        folder_path: Path to the folder containing images
        supported_extensions: List of supported file extensions (default: common image formats)
    
    Returns:
        List of full paths to image files, sorted by filename
    """
    if supported_extensions is None:
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    image_files = []
    for ext in supported_extensions:
        # Case insensitive search
        pattern = f"*{ext}"
        image_files.extend(folder_path.glob(pattern))
        image_files.extend(folder_path.glob(pattern.upper()))
    
    # Sort by filename to ensure consistent ordering
    image_files = sorted([str(f) for f in image_files])
    
    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")
    
    print(f"Found {len(image_files)} image files in {folder_path}")
    for i, img_file in enumerate(image_files[:5]):  # Show first 5 files
        print(f"  {i+1}: {Path(img_file).name}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more files")
    
    return image_files

def convert_glb_to_ply(glb_path, ply_path=None):
    """
    Convert GLB file to PLY format.
    
    Args:
        glb_path: Path to input GLB file
        ply_path: Path for output PLY file (optional, will auto-generate if None)
    
    Returns:
        Path to the created PLY file
    """
    if not os.path.exists(glb_path):
        raise ValueError(f"GLB file does not exist: {glb_path}")
    
    if ply_path is None:
        ply_path = glb_path.replace('.glb', '.ply')
    
    try:
        print(f"Converting GLB to PLY: {glb_path} -> {ply_path}")
        
        # Load the GLB scene
        scene = trimesh.load(glb_path)
        
        # If it's a scene with multiple geometries, merge them
        if hasattr(scene, 'geometry'):
            # Extract all mesh/pointcloud geometries
            geometries = []
            for geom in scene.geometry.values():
                if hasattr(geom, 'vertices'):
                    geometries.append(geom)
            
            if len(geometries) == 1:
                # Single geometry
                geometry = geometries[0]
            elif len(geometries) > 1:
                # Multiple geometries - merge them
                if all(hasattr(g, 'faces') for g in geometries):
                    # All are meshes
                    geometry = trimesh.util.concatenate(geometries)
                else:
                    # Mix of meshes and point clouds, or all point clouds
                    all_vertices = []
                    all_colors = []
                    for geom in geometries:
                        if hasattr(geom, 'vertices'):
                            all_vertices.append(geom.vertices)
                            if hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors'):
                                all_colors.append(geom.visual.vertex_colors)
                            else:
                                # Default color if no colors
                                all_colors.append(np.full((len(geom.vertices), 4), [128, 128, 128, 255], dtype=np.uint8))
                    
                    vertices = np.vstack(all_vertices)
                    colors = np.vstack(all_colors)
                    geometry = trimesh.PointCloud(vertices=vertices, colors=colors)
            else:
                raise ValueError("No valid geometries found in GLB file")
        else:
            # Single geometry
            geometry = scene
        
        # Export to PLY
        geometry.export(ply_path)
        
        if os.path.exists(ply_path):
            file_size = os.path.getsize(ply_path)
            print(f"PLY file successfully created: {ply_path} (size: {file_size} bytes)")
            return ply_path
        else:
            raise ValueError("PLY file was not created")
            
    except Exception as e:
        print(f"Error converting GLB to PLY: {e}")
        return None

def local_get_reconstructed_scene(filelist, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, cam_size,
                                  shared_intrinsics, image_size=512, **kw):
    """Local function to get reconstructed scene with default parameters."""
    lr1 = 0.05
    niter1 = 500
    lr2 = 0.014
    niter2 = 200
    optim_level = 'refine'
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    silent = True
    gradio_delete_cache = 7200

    if len(filelist) < 5:
        scenegraph_type = 'complete'
        winsize = 1
    else:
        scenegraph_type = 'logwin'
        half_size = math.ceil((len(filelist) - 1) / 2)
        max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        winsize = min(5, max_winsize)
        
    refid = 0
    win_cyclic = False
    
    # Get temp directory
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load model
    weights_path = os.path.join(MAST3R_ROOT, "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    scene_state, outfile = get_reconstructed_scene(temp_dir, gradio_delete_cache, model, device, silent, image_size, None,
                                                   filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                                   as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                                                   win_cyclic, refid, TSDF_thresh=0, shared_intrinsics=shared_intrinsics, **kw)
    
    outfile_path = scene_state.outfile_name  # cache the path
    scene_state.should_delete = False        # prevent deletion
    filestate = FileState(outfile_path)
    del scene_state
    return filestate, outfile

def run_mast3r(image_path, output_dir, matching_conf_thr=1.0, min_conf_thr=0.5, 
               cam_size=0.2, as_pointcloud=True, shared_intrinsics=True, 
               max_images=None, convert_to_ply=True, image_size=512, 
               resize_images=False, **kw):
    """
    Run 3D reconstruction from images using MAST3R.
    
    Args:
        image_path: Path to folder containing images or list of image files
        output_dir: Output directory for results
        matching_conf_thr: Confidence threshold for matching
        min_conf_thr: Minimum confidence threshold
        cam_size: Camera visualization size
        as_pointcloud: Whether to export as pointcloud (True) or mesh (False)
        shared_intrinsics: Whether to use shared camera intrinsics
        max_images: Maximum number of images to process (None for all)
        convert_to_ply: Whether to also create a PLY file
        image_size: Size to resize images to (default: 512)
        resize_images: Whether to resize images to 512x512 before processing (default: False)
        **kw: Additional keyword arguments
    
    Returns:
        Tuple of (glb_path, ply_path)
    """
    if not MAST3R_AVAILABLE:
        raise ImportError("MAST3R dependencies are not available. Please install MAST3R first.")
    
    # Handle input path
    if isinstance(image_path, str):
        if os.path.isdir(image_path):
            # Directory of images
            if resize_images:
                print(f"Resizing images in {image_path} to {image_size}x{image_size}...")
                filelist = resize_images_in_folder(image_path, target_size=image_size)
            else:
                filelist = get_image_files_from_folder(image_path)
        else:
            # Single image file
            if resize_images:
                print(f"Resizing image {image_path} to {image_size}x{image_size}...")
                resized_path = resize_image_to_512(image_path)
                filelist = [resized_path]
            else:
                filelist = [image_path]
    else:
        # List of image files
        if resize_images:
            print(f"Resizing {len(image_path)} images to {image_size}x{image_size}...")
            resized_files = []
            for img_file in image_path:
                resized_path = resize_image_to_512(img_file)
                resized_files.append(resized_path)
            filelist = resized_files
        else:
            filelist = image_path
    
    # Limit number of images if specified
    if max_images is not None and len(filelist) > max_images:
        print(f"Limiting to first {max_images} images out of {len(filelist)} found")
        filelist = filelist[:max_images]
    
    print(f"Processing {len(filelist)} images with size {image_size}x{image_size}")
    
    # Run reconstruction
    filestate, glb_path = local_get_reconstructed_scene(
        filelist, min_conf_thr, matching_conf_thr, 
        as_pointcloud, cam_size, shared_intrinsics, image_size=image_size, **kw
    )
    
    # Convert to PLY if requested
    ply_path = None
    if convert_to_ply and glb_path and os.path.exists(glb_path):
        ply_path = convert_glb_to_ply(glb_path)
    
    return glb_path, ply_path

def resize_image_to_512(image_path, output_path=None):
    """
    Resize an image to 512x512 pixels.
    
    Args:
        image_path: Path to input image
        output_path: Path for resized image (optional, will overwrite if None)
    
    Returns:
        Path to the resized image
    """
    if output_path is None:
        output_path = image_path
    
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 512x512
            resized_img = img.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Save the resized image
            resized_img.save(output_path, quality=95)
            
            print(f"Resized {image_path} to 512x512 -> {output_path}")
            return output_path
            
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return image_path

def resize_images_in_folder(folder_path, target_size=512):
    """
    Resize all images in a folder to the target size.
    
    Args:
        folder_path: Path to folder containing images
        target_size: Target size for both width and height (default: 512)
    
    Returns:
        List of paths to resized images
    """
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    resized_paths = []
    
    for ext in supported_extensions:
        # Case insensitive search
        pattern = f"*{ext}"
        image_files = list(folder_path.glob(pattern)) + list(folder_path.glob(pattern.upper()))
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    # Check if resizing is needed
                    if img.size != (target_size, target_size):
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize to target size
                        resized_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        
                        # Save the resized image (overwrite original)
                        resized_img.save(img_file, quality=95)
                        print(f"Resized {img_file.name} to {target_size}x{target_size}")
                    
                    resized_paths.append(str(img_file))
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    return sorted(resized_paths)

if __name__ == "__main__":
    # Example usage
    image_folder = "/home/t/Desktop/color"  # Change this to your folder path
    output_dir = "/home/t/Desktop/output"
    
    try:
        glb_path, ply_path = run_mast3r(
            image_folder,
            output_dir,
            matching_conf_thr=1.0,
            min_conf_thr=0.5,
            as_pointcloud=True,
            convert_to_ply=True
        )
        
        print(f"\n=== MAST3R RECONSTRUCTION COMPLETE ===")
        print(f"GLB file: {glb_path}")
        if ply_path:
            print(f"PLY file: {ply_path}")
        
    except Exception as e:
        print(f"Error during MAST3R reconstruction: {e}")
        import traceback
        traceback.print_exc() 