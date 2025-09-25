import os
import sys
import cv2
import torch
import numpy as np
import glob
import shutil
import time
import gc
from datetime import datetime
from pathlib import Path
import tempfile

# Add VGGT to path
VGGT_ROOT = "vggt/"
if VGGT_ROOT not in sys.path:
    sys.path.insert(0, VGGT_ROOT)

# Import VGGT dependencies
try:
    from visual_util import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    VGGT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VGGT dependencies not available: {e}")
    print("VGGT reconstruction will not be available")
    print(f"Make sure VGGT is installed at: {VGGT_ROOT}")
    VGGT_AVAILABLE = False

def setup_vggt_model():
    """Initialize and load the VGGT model."""
    print("🔄 Initializing and loading VGGT model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  Warning: CUDA not available, using CPU (this will be much slower)")
    
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    
    print(f"✅ VGGT model loaded successfully on {device}")
    return model, device

def extract_frames_from_video(video_path, output_dir, fps=1):
    """Extract frames from video at specified FPS."""
    print(f"📹 Extracting frames from video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
    
    cap.release()
    print(f"✅ Extracted {saved_count} frames from video")
    return saved_count

def prepare_images_for_vggt(input_path, temp_dir):
    """Prepare images for VGGT processing - handle single image, directory, or video."""
    print(f"📁 Preparing images from: {input_path}")
    
    input_path = Path(input_path)
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    if input_path.is_file():
        # Single file - could be image or video
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video file
            extract_frames_from_video(str(input_path), images_dir)
        else:
            # Single image
            shutil.copy2(str(input_path), os.path.join(images_dir, input_path.name))
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {input_path}")
        
        for img_file in sorted(image_files):
            shutil.copy2(str(img_file), os.path.join(images_dir, img_file.name))
        
        print(f"✅ Copied {len(image_files)} images")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    return images_dir

def run_vggt_inference(model, device, images_dir):
    """Run VGGT inference on the prepared images."""
    print("🧠 Running VGGT inference...")
    
    # Load and preprocess images
    image_names = glob.glob(os.path.join(images_dir, "*"))
    image_names = sorted(image_names)
    
    if len(image_names) == 0:
        raise ValueError("No images found for processing")
    
    print(f"📸 Processing {len(image_names)} images")
    images = load_and_preprocess_images(image_names).to(device)
    print(f"📐 Preprocessed images shape: {images.shape}")
    
    # Run inference
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    # Convert pose encoding to extrinsic and intrinsic matrices
    print("🔄 Converting pose encoding to camera matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None
    
    # Generate world points from depth map
    print("🌍 Computing world points from depth map...")
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Clean up
    torch.cuda.empty_cache()
    return predictions

def save_vggt_results(predictions, output_dir, conf_threshold=3.0, show_cameras=True):
    """Save the VGGT reconstruction results."""
    print("💾 Saving VGGT reconstruction results...")
    
    # Save raw predictions
    predictions_path = os.path.join(output_dir, "predictions.npz")
    np.savez(predictions_path, **predictions)
    print(f"📊 Raw predictions saved to: {predictions_path}")
    
    # Generate GLB file
    glb_filename = f"reconstruction_conf{conf_threshold}_cam{show_cameras}.glb"
    glb_path = os.path.join(output_dir, glb_filename)
    
    print("🎨 Generating 3D scene...")
    glb_scene = predictions_to_glb(
        predictions,
        conf_thres=conf_threshold,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=show_cameras,
        mask_sky=False,
        target_dir=output_dir,
        prediction_mode="Pointmap Regression",
    )
    
    glb_scene.export(file_obj=glb_path)
    print(f"🎯 3D reconstruction saved to: {glb_path}")
    
    return glb_path

def convert_glb_to_ply(glb_path, ply_path=None):
    """
    Convert GLB file to PLY point cloud format.
    
    Args:
        glb_path: Path to input GLB file
        ply_path: Path for output PLY file (optional)
    
    Returns:
        Path to the PLY file, or None if conversion failed
    """
    try:
        import trimesh
        
        if ply_path is None:
            ply_path = glb_path.replace('.glb', '.ply')
        
        # Load the GLB file
        scene = trimesh.load(glb_path)
        
        # Extract vertices from all meshes in the scene
        vertices = []
        colors = []
        
        if hasattr(scene, 'geometry'):
            # Scene with multiple meshes
            for mesh in scene.geometry.values():
                if hasattr(mesh, 'vertices'):
                    vertices.append(mesh.vertices)
                    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                        colors.append(mesh.visual.vertex_colors[:, :3])  # RGB only
        elif hasattr(scene, 'vertices'):
            # Single mesh
            vertices.append(scene.vertices)
            if hasattr(scene, 'visual') and hasattr(scene.visual, 'vertex_colors'):
                colors.append(scene.visual.vertex_colors[:, :3])  # RGB only
        
        if not vertices:
            print("Warning: No vertices found in GLB file")
            return None
        
        # Combine all vertices
        all_vertices = np.vstack(vertices)
        
        # Create point cloud
        pcd = trimesh.PointCloud(vertices=all_vertices)
        
        # Add colors if available
        if colors:
            all_colors = np.vstack(colors)
            if len(all_colors) == len(all_vertices):
                pcd.colors = all_colors
        
        # Save as PLY
        pcd.export(ply_path)
        print(f"✅ Converted GLB to PLY: {ply_path}")
        return ply_path
        
    except Exception as e:
        print(f"Error converting GLB to PLY: {e}")
        return None

def run_vggt(image_path, output_dir, conf_threshold=3.0, show_cameras=True, 
             fps=1.0, convert_to_ply=True, **kw):
    """
    Run 3D reconstruction from images using VGGT.
    
    Args:
        image_path: Path to folder containing images, video file, or single image
        output_dir: Output directory for results
        conf_threshold: Confidence threshold for point filtering (default: 3.0)
        show_cameras: Whether to show cameras in the 3D scene (default: True)
        fps: Frames per second to extract from video (default: 1.0)
        convert_to_ply: Whether to also create a PLY file (default: True)
        **kw: Additional keyword arguments
    
    Returns:
        Tuple of (glb_path, ply_path)
    """
    if not VGGT_AVAILABLE:
        raise ImportError("VGGT dependencies are not available. Please install VGGT first.")
    
    # Create temporary directory for processing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_dir = os.path.join(output_dir, f"temp_vggt_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("🚀 Starting VGGT 3D Reconstruction")
        print("=" * 50)
        
        # Setup model
        model, device = setup_vggt_model()
        
        # Prepare images
        images_dir = prepare_images_for_vggt(image_path, temp_dir)
        
        # Run inference
        start_time = time.time()
        predictions = run_vggt_inference(model, device, images_dir)
        
        # Save results
        glb_path = save_vggt_results(
            predictions, 
            output_dir, 
            conf_threshold=conf_threshold,
            show_cameras=show_cameras
        )
        
        end_time = time.time()
        
        print(f"⏱️  VGGT processing time: {end_time - start_time:.2f} seconds")
        print(f"🎯 3D reconstruction: {glb_path}")
        
        # Convert to PLY if requested
        ply_path = None
        if convert_to_ply and glb_path and os.path.exists(glb_path):
            ply_path = convert_glb_to_ply(glb_path)
            if ply_path:
                print(f"📊 Point cloud saved to: {ply_path}")
        
        return glb_path, ply_path
        
    except Exception as e:
        print(f"❌ Error during VGGT processing: {e}")
        raise e
    
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            print(f"🧹 Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
