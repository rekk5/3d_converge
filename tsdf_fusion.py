import os
import glob
import time
import argparse
import yaml

import open3d as o3d
import numpy as np

from scipy.spatial.transform import Rotation


class TSDFFusion:
    """
    A class to perform TSDF fusion for 3D reconstruction from RGB-D images and camera poses.
    """
    
    def __init__(self, data_path=None, voxel_length=0.01, sdf_trunc=0.04):
        """
        Initialize the TSDF fusion processor.
        
        Args:
            data_path (str): Path to the data directory containing color, depth, poses, and calibration
            voxel_length (float): Length of each voxel in meters
            sdf_trunc (float): Truncation distance for the signed distance function
        """
        self.data_path = data_path
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        
        # Paths for required data
        if data_path:
            self.color_path = os.path.join(data_path, "color")
            self.depth_path = os.path.join(data_path, "depth_render")
            self.poses_path = os.path.join(data_path, "poses", "poses_color.txt")
            self.meshs_path = os.path.join(data_path, "recon")
            self.calib_path = os.path.join(data_path, "calibration", "calib_color.yaml")
        
        # Data containers
        self.color_list = []
        self.depth_list = []
        self.cam_poses = None
        self.timestamps = None
        self.intrinsic = None
        
        # Results
        self.pcd = None
        self.mesh = None
    
    def read_trajectory(self, fpath):
        """
        Reads a trajectory from a file and returns the poses and timestamps.
        
        Args:
            fpath (str): The file path of the trajectory file.
            
        Returns:
            tuple: (poses, timestamps) where poses is an array of shape (N, 4, 4)
                  and timestamps is an array of shape (N,)
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
            
            for i, (qt, t) in enumerate(zip(qts, tvecs)):
                rot = Rotation.from_quat(qt)
                R = rot.as_matrix()
                poses[i, :3, :3] = R
                poses[i, :3, 3] = t
                
            return poses, timestamps
        
        return None, None
    
    def validate_input_data(self):
        """
        Validates that the provided directory contains all necessary data for TSDF fusion.
        
        Returns:
            bool: True if the data is valid, False otherwise
        """
        # Check if data directory exists
        if not os.path.isdir(self.data_path):
            print(f"Error: Data directory not found: {self.data_path}")
            return False
        
        # Check for required subdirectories
        required_dirs = ["color", "depth_render", "poses", "calibration"]
        for dir_name in required_dirs:
            dir_path = os.path.join(self.data_path, dir_name)
            if not os.path.isdir(dir_path):
                print(f"Error: Required directory not found: {dir_path}")
                return False
        
        # Check for RGB images
        self.color_list = glob.glob(os.path.join(self.color_path, "*.jpg")) + glob.glob(os.path.join(self.color_path, "*.png"))
        self.color_list.sort()
        if not self.color_list:
            print(f"Error: No RGB images found in {self.color_path}")
            return False
        
        # Check for depth images
        self.depth_list = glob.glob(os.path.join(self.depth_path, "*.png"))
        self.depth_list.sort()
        if not self.depth_list:
            print(f"Error: No depth images found in {self.depth_path}")
            return False
        
        # Check if number of RGB and depth images match
        if len(self.color_list) != len(self.depth_list):
            print(f"Warning: Number of RGB images ({len(self.color_list)}) does not match number of depth images ({len(self.depth_list)})")
        
        # Check for pose file
        if not os.path.isfile(self.poses_path):
            print(f"Error: Pose file not found: {self.poses_path}")
            return False
        
        # Check for calibration file
        if not os.path.isfile(self.calib_path):
            print(f"Error: Calibration file not found: {self.calib_path}")
            return False
        
        # Verify pose file format
        try:
            poses_data = np.loadtxt(self.poses_path, delimiter=' ')
            if poses_data.shape[0] < 1:
                print(f"Error: Pose file is empty: {self.poses_path}")
                return False
            if poses_data.shape[1] != 8:  # timestamp + translation + quaternion
                print(f"Error: Pose file has incorrect format. Expected 8 columns, got {poses_data.shape[1]}")
                return False
        except Exception as e:
            print(f"Error reading pose file: {e}")
            return False
        
        # Verify calibration file format
        try:
            with open(self.calib_path, 'r') as f:
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
        
        print(f"TSDF input data validated: {len(self.color_list)} RGB images, {len(self.depth_list)} depth images")
        return True
    
    def load_data(self):
        """
        Load all required data: camera poses, calibration parameters.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.meshs_path):
            os.makedirs(self.meshs_path)
            
        # Load the camera poses
        self.cam_poses, self.timestamps = self.read_trajectory(self.poses_path)
        print("cam_poses.shape: ", self.cam_poses.shape)
        print("cam_poses[0]: ", self.cam_poses[0])
        
        # Load calibration data from YAML file
        try:
            with open(self.calib_path, 'r') as f:
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
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                image_width,
                image_height,
                intrinsic_matrix[0, 0],  # fx
                intrinsic_matrix[1, 1],  # fy
                intrinsic_matrix[0, 2],  # cx
                intrinsic_matrix[1, 2]   # cy
            )
            print("Check intrinsic matrix:", self.intrinsic.intrinsic_matrix)
            return True
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def perform_fusion(self):
        """
        Performs TSDF fusion on the sequence of RGB-D frames.
        
        Returns:
            bool: True if fusion was successful, False otherwise
        """
        # Create TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        total_time = 0.0
        
        # Process each frame
        for i in range(len(self.color_list)):
            start_time = time.perf_counter()
            print(f"Integrating frame {i+1}/{len(self.color_list)}")
            
            try:
                color = o3d.io.read_image(self.color_list[i])
                depth = o3d.io.read_image(self.depth_list[i])
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
                
                print("rgbd: ", rgbd)
                print("np.linalg.inv(cam_poses[i]): ", np.linalg.inv(self.cam_poses[i]))
                volume.integrate(rgbd, self.intrinsic, np.linalg.inv(self.cam_poses[i]))
                
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
                print(f"Processing time for frame {i+1}: {elapsed_time * 1000:.2f} ms")
                
            except Exception as e:
                print(f"Error processing frame {i+1}: {e}")
                continue
        
        # Calculate average processing time
        if len(self.color_list) > 0:
            average_time = (total_time / len(self.color_list)) * 1000  # in milliseconds
            print(f"Average processing time for {len(self.color_list)} frames: {average_time:.2f} ms")
        
        # Extract results
        try:
            self.mesh = volume.extract_triangle_mesh()
            self.mesh.compute_vertex_normals()
            self.pcd = volume.extract_point_cloud()
            return True
        except Exception as e:
            print(f"Error extracting results: {e}")
            return False
    
    def save_results(self, save_pcd=True, save_mesh=True):
        """
        Save the point cloud and/or mesh to files.
        
        Args:
            save_pcd (bool): Whether to save the point cloud
            save_mesh (bool): Whether to save the mesh
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not os.path.exists(self.meshs_path):
            os.makedirs(self.meshs_path)
            
        try:
            if save_pcd and self.pcd is not None:
                o3d.io.write_point_cloud(os.path.join(self.meshs_path, "point_cloud.ply"), self.pcd)
                print(f"Point cloud saved to {os.path.join(self.meshs_path, 'point_cloud.ply')}")
                
            if save_mesh and self.mesh is not None:
                o3d.io.write_triangle_mesh(os.path.join(self.meshs_path, "mesh.ply"), self.mesh)
                print(f"Mesh saved to {os.path.join(self.meshs_path, 'mesh.ply')}")
                
            return True
        
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def run(self, save_pcd=True, save_mesh=True):
        """
        Run the complete TSDF fusion pipeline.
        
        Args:
            save_pcd (bool): Whether to save the point cloud
            save_mesh (bool): Whether to save the mesh
            
        Returns:
            tuple: (point_cloud, mesh) the resulting geometries
        """
        # Check if we have all the required data
        if not self.validate_input_data():
            print("Error: Invalid input data for TSDF fusion")
            return None, None
        
        # Load data
        if not self.load_data():
            print("Error: Failed to load required data for TSDF fusion")
            return None, None
        
        # Perform fusion
        if not self.perform_fusion():
            print("Error: TSDF fusion failed")
            return None, None
        
        # Save results
        if save_pcd or save_mesh:
            self.save_results(save_pcd, save_mesh)
        
        return self.pcd, self.mesh


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D reconstruction using TSDF fusion')
    parser.add_argument('--data_path', type=str, help='Path to data directory')
    parser.add_argument('--voxel_length', type=float, default=0.01, help='Voxel length for TSDF fusion')
    parser.add_argument('--sdf_trunc', type=float, default=0.04, help='Truncation distance for the signed distance function')
    parser.add_argument('--save_pcd', action='store_true', help='Save the point cloud')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Use default data path if not provided
    if not args.data_path:
        args.data_path = "/home/t/Desktop/cwcorg"
    
    # Create TSDF fusion processor
    tsdf = TSDFFusion(
        data_path=args.data_path,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc
    )
    
    # Run the TSDF fusion pipeline
    start_time = time.perf_counter()
    pcd, mesh = tsdf.run(save_pcd=args.save_pcd, save_mesh=args.save_mesh)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    print(f"Done ☕️☕️☕️ Total processing time: {total_time:.2f} seconds")

