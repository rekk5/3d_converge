import os
import glob
import time
import argparse
import yaml

import open3d as o3d
import numpy as np

from scipy.spatial.transform import Rotation



def read_trajectory(fpath):
    """
    Reads a trajectory from a file and returns the poses and timestamps.
    Parameters:
    - fpath (str): The file path of the trajectory file.
    Returns:
    - poses (ndarray): An array of shape (N, 4, 4) representing the poses of the trajectory.
      Each pose is a 4x4 transformation matrix.
    - timestamps (ndarray): An array of shape (N,) representing the timestamps of the poses.
    Note:
    - The trajectory file should be in a specific format where each line contains the following information:
      timestamp t_x t_y t_z q_x q_y q_z q_w
      - timestamp: The timestamp of the pose.
      - t_x, t_y, t_z: The translation components of the pose.
      - q_x, q_y, q_z, q_w: The quaternion components of the pose.
    """
    data = np.loadtxt(fpath, delimiter=' ')
    
    N = data.shape[0]
    eye = np.eye(4, dtype=np.float32)
    eye = np.expand_dims(eye, axis=0)
    poses = np.repeat(eye, N, axis=0)
    
    if data.shape[1] == 8:
        timestamps = data[:, 0]
        qts = data[:, 4:]
        tvecs = data[:, 1:4]
        
        for i, (qt, t) in enumerate(zip(qts,tvecs)):
            rot = Rotation.from_quat(qt)
            R = rot.as_matrix()
            poses[i, :3, :3] = R
            poses[i, :3, 3] = t
            
        return poses, timestamps

def perform_tsdf_fusion(rgb_files, depth_files, intrinsic, extrinsic_list, voxel_length=0.007125, sdf_trunc=0.04):   
    """
    Performs TSDF fusion on a sequence of RGB-D frames.

    Args:
        rgb_files (List[str]): List of file paths to RGB images.
        depth_files (List[str]): List of file paths to depth images.
        intrinsic (numpy.ndarray): Intrinsic camera matrix.
        extrinsic_list (List[numpy.ndarray]): List of extrinsic camera matrices.
        voxel_length (float, optional): Length of each voxel in meters. Defaults to 0.007.
        sdf_trunc (float, optional): Truncation distance for the signed distance function. Defaults to 0.04.

    Returns:
        Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]: A tuple containing the fused point cloud and the extracted triangle mesh.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    total_time = 0.0

    for i in range(len(rgb_files)):
        start_time = time.perf_counter()
        print(f"Integrating frame {i+1}/{len(rgb_files)}")
        color = o3d.io.read_image(rgb_files[i])
        depth = o3d.io.read_image(depth_files[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        print("rgbd: ", rgbd)
        print("np.linalg.inv(cam_poses[i]): ", np.linalg.inv(extrinsic_list[i]))
        volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic_list[i]))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Processing time for frame {i+1}: {elapsed_time * 1000:.2f} ms")

    average_time = (total_time / len(rgb_files)) * 1000  # in milliseconds
    print(f"Average processing time for {len(rgb_files)} frames: {average_time:.2f} ms")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()
    return pcd, mesh

def validate_tsdf_input_data(data_path):
    """
    Validates that the provided directory contains all necessary data for TSDF fusion.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        bool: True if the data is valid, False otherwise
    """
    import os
    import glob
    
    # Check if data directory exists
    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found: {data_path}")
        return False
    
    # Check for required subdirectories
    required_dirs = ["color", "depth_render", "poses", "calibration"]
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.isdir(dir_path):
            print(f"Error: Required directory not found: {dir_path}")
            return False
    
    # Check for RGB images
    color_path = os.path.join(data_path, "color")
    color_files = glob.glob(os.path.join(color_path, "*.jpg")) + glob.glob(os.path.join(color_path, "*.png"))
    if not color_files:
        print(f"Error: No RGB images found in {color_path}")
        return False
    
    # Check for depth images
    depth_path = os.path.join(data_path, "depth_render")
    depth_files = glob.glob(os.path.join(depth_path, "*.png"))
    if not depth_files:
        print(f"Error: No depth images found in {depth_path}")
        return False
    
    # Check if number of RGB and depth images match
    if len(color_files) != len(depth_files):
        print(f"Warning: Number of RGB images ({len(color_files)}) does not match number of depth images ({len(depth_files)})")
    
    # Check for pose file
    poses_path = os.path.join(data_path, "poses", "poses_color.txt")
    if not os.path.isfile(poses_path):
        print(f"Error: Pose file not found: {poses_path}")
        return False
    
    # Check for calibration file
    calib_path = os.path.join(data_path, "calibration", "calib_color.yaml")
    if not os.path.isfile(calib_path):
        print(f"Error: Calibration file not found: {calib_path}")
        return False
    
    # Verify pose file format
    try:
        poses_data = np.loadtxt(poses_path, delimiter=' ')
        if poses_data.shape[0] < 1:
            print(f"Error: Pose file is empty: {poses_path}")
            return False
        if poses_data.shape[1] != 8:  # timestamp + translation + quaternion
            print(f"Error: Pose file has incorrect format. Expected 8 columns, got {poses_data.shape[1]}")
            return False
    except Exception as e:
        print(f"Error reading pose file: {e}")
        return False
    
    # Verify calibration file format
    try:
        with open(calib_path, 'r') as f:
            first_line = f.readline()
            if ':' in first_line:
                first_line = first_line.replace(':', ' ')
            calib_data = yaml.safe_load(first_line + f.read())
        
        required_calib_fields = ["image_width", "image_height", "projection_matrix"]
        for field in required_calib_fields:
            if field not in calib_data:
                print(f"Error: Calibration file missing required field: {field}")
                return False
    except Exception as e:
        print(f"Error reading calibration file: {e}")
        return False
    
    print(f"TSDF input data validated: {len(color_files)} RGB images, {len(depth_files)} depth images")
    return True

def main(args):
    # Set the data paths
    args.data_path = f"/home/t/Desktop/cwcorg"
    
    # Validate input data
    if not validate_tsdf_input_data(args.data_path):
        print("Error: Invalid input data for TSDF fusion")
        return
    
    color_path = os.path.join(args.data_path, "color")
    depth_path = os.path.join(args.data_path, "depth_render")
    poses_path = os.path.join(args.data_path, "poses", "poses_color.txt")
    meshs_path = os.path.join(args.data_path, "recon")
    calib_path = os.path.join(args.data_path, "calibration", "calib_color.yaml")

    # Create the meshs directory if it does not exist
    if not os.path.exists(meshs_path):
        os.makedirs(meshs_path)

    # Load the color and depth images name list
    color_list = glob.glob(os.path.join(color_path, "*.jpg"))
    color_list.sort()
    depth_list = glob.glob(os.path.join(depth_path, "*.png"))
    depth_list.sort()

    # Load the camera poses   nn = 10
    std_multiplier = 2.0
    cam_poses, _ = read_trajectory(poses_path)
    print("cam_poses.shape: ", cam_poses.shape)
    print("cam_poses[0]: ", cam_poses[0])

    # Load calibration data from YAML file
    with open(calib_path, 'r') as f:
        first_line = f.readline()
        if ':' in first_line:
            print("Found ':' in first line")
            first_line = first_line.replace(':', ' ')
        calib_data = yaml.safe_load(first_line + f.read())
    
    # Extract calibration data
    image_width = calib_data['image_width']
    image_height = calib_data['image_height']
    projection_matrix = np.array(calib_data['projection_matrix']['data']).reshape(3, 4)
    
    # Check calibration data
    print("Image width:", image_width)
    print("Image height:", image_height)
    print("Projection matrix:", projection_matrix)

    # Extract intrinsic matrix from projection matrix
    intrinsic_matrix = projection_matrix[:3, :3]

    # Create PinholeCameraIntrinsic object using loaded calibration data
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width,
        image_height,
        intrinsic_matrix[0, 0],  # fx
        intrinsic_matrix[1, 1],  # fy
        intrinsic_matrix[0, 2],  # cx
        intrinsic_matrix[1, 2]   # cy
    )
    print("Check intrinsic matrix:", intrinsic.intrinsic_matrix)

    # Perform TSDF fusion
    pcd, mesh = perform_tsdf_fusion(color_list, depth_list, intrinsic, cam_poses, voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc)

    # Save the point cloud and mesh
    if args.save_pcd:
        o3d.io.write_point_cloud(os.path.join(meshs_path, f"point_cloud.ply"), pcd)
    if args.save_mesh:
        o3d.io.write_triangle_mesh(os.path.join(meshs_path, f"mesh.ply"), mesh)

if __name__ == "__main__":
    # Parse command line arguments
    def parse_arguments():
        parser = argparse.ArgumentParser(description='3D reconstruction using TSDF fusion')
        parser.add_argument('--data_path', type=str, help='Path to data directory')
        parser.add_argument('--voxel_length', type=float, default=0.01, help='Voxel length for TSDF fusion')
        parser.add_argument('--sdf_trunc', type=float, default=0.04, help='Truncation distance for the signed distance function')
        parser.add_argument('--save_pcd', action='store_true', help='Save the point cloud')
        parser.add_argument('--save_mesh', action='store_true', help='Save the mesh')
        return parser.parse_args()

    args = parse_arguments()
    start_time = time.perf_counter()
    main(args)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Done ☕️☕️☕️ Total processing time: {total_time:.2f} seconds")

