import os
import argparse
import sys
import numpy as np
import open3d as o3d

from colmap_recon import run_colmap
from tsdf_fusion import TSDFFusion
from segmentation import PointCloudSegmentation, validate_point_cloud_file
from diffraction import DiffractionEdgeDetector, validate_segmented_point_cloud

def parse_arguments():
    parser = argparse.ArgumentParser(description='3D Reconstruction Pipeline')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, help='Path to the input images')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--recon_method', type=str, choices=['colmap', 'tsdf', 'none'], default='colmap', 
                        help='Reconstruction method to use (colmap, tsdf, or none to skip reconstruction)')
    parser.add_argument('--point_cloud_path', type=str, help='Path to existing point cloud (skip reconstruction)')
    
    # TSDF specific parameters
    parser.add_argument('--depth_path', type=str, help='Path to depth images (required for TSDF)')
    parser.add_argument('--voxel_length', type=float, default=0.01, help='Voxel length for TSDF fusion')
    parser.add_argument('--sdf_trunc', type=float, default=0.04, help='Truncation distance for TSDF')
    
    # Point cloud segmentation parameters
    parser.add_argument('--skip_segmentation', action='store_true', help='Skip the segmentation phase')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size for downsampling')
    parser.add_argument('--max_planes', type=int, default=5, help='Maximum number of planes to detect')
    parser.add_argument('--distance_threshold', type=float, default=0.02, help='Distance threshold for plane detection')
    
    # Diffraction edge parameters
    parser.add_argument('--skip_diffraction', action='store_true', help='Skip the diffraction edge detection phase')
    parser.add_argument('--angle_threshold', type=float, default=45.0, help='Angle threshold for edge detection (degrees)')
    parser.add_argument('--min_edge_length', type=float, default=0.1, help='Minimum edge length to consider')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: 3D Reconstruction (optional)
    point_cloud_path = args.point_cloud_path
    
    if args.recon_method != 'none' and point_cloud_path is None:
        if args.image_path is None:
            print("Error: --image_path is required for reconstruction")
            sys.exit(1)
            
        if args.recon_method == 'colmap':
            print("===== PHASE 1: COLMAP RECONSTRUCTION =====")
            print(f"Starting COLMAP reconstruction with images from: {args.image_path}")
            
            # Create output directory for COLMAP
            colmap_output_dir = os.path.join(args.output_dir, 'colmap')
            os.makedirs(colmap_output_dir, exist_ok=True)
            
            # Run COLMAP reconstruction
            point_cloud_path = run_colmap(
                image_path=args.image_path,
                database_path=os.path.join(colmap_output_dir, 'database.db'),
                sparse_output_path=os.path.join(colmap_output_dir, 'sparse'),
                dense_output_path=os.path.join(colmap_output_dir, 'dense')
            )
            
            print(f"COLMAP reconstruction completed. Point cloud saved to: {point_cloud_path}")
        
        elif args.recon_method == 'tsdf':
            print("===== PHASE 1: TSDF FUSION RECONSTRUCTION =====")
            
            # Validate required paths for TSDF
            if not args.depth_path:
                print("Error: TSDF fusion requires depth images. Please provide --depth_path")
                sys.exit(1)
            
            # Create required directory structure for TSDF
            tsdf_data_path = os.path.join(args.output_dir, 'tsdf_data')
            os.makedirs(tsdf_data_path, exist_ok=True)
            
            # Create required subdirectories
            color_dir = os.path.join(tsdf_data_path, 'color')
            depth_dir = os.path.join(tsdf_data_path, 'depth_render')
            poses_dir = os.path.join(tsdf_data_path, 'poses')
            calib_dir = os.path.join(tsdf_data_path, 'calibration')
            
            for directory in [color_dir, depth_dir, poses_dir, calib_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # Copy/link input data to the expected structure
            # Note: You'll need to implement these functions based on your data format
            try:
                # Setup color images
                setup_color_images(args.image_path, color_dir)
                
                # Setup depth images
                setup_depth_images(args.depth_path, depth_dir)
                
                # Setup camera poses
                setup_camera_poses(args.image_path, os.path.join(poses_dir, 'poses_color.txt'))
                
                # Setup calibration
                setup_camera_calibration(args.image_path, os.path.join(calib_dir, 'calib_color.yaml'))
                
                # Run TSDF fusion
                tsdf = TSDFFusion(
                    data_path=tsdf_data_path,
                    voxel_length=args.voxel_length,
                    sdf_trunc=args.sdf_trunc
                )
                
                pcd, mesh = tsdf.run(save_pcd=True, save_mesh=True)
                
                if pcd is not None:
                    point_cloud_path = os.path.join(tsdf_data_path, 'recon', 'point_cloud.ply')
                    print(f"TSDF fusion completed. Point cloud saved to: {point_cloud_path}")
                else:
                    print("Error: TSDF fusion failed to generate a point cloud")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"Error during TSDF fusion setup: {e}")
                sys.exit(1)
    
    elif point_cloud_path is None:
        print("Error: You must either specify a reconstruction method or provide an existing point cloud path")
        sys.exit(1)
    else:
        print(f"Skipping reconstruction, using existing point cloud: {point_cloud_path}")
    
    # Phase 2: Point Cloud Segmentation (optional)
    segmentation_output_path = None
    if not args.skip_segmentation:
        try:
            print("\n===== PHASE 2: POINT CLOUD SEGMENTATION =====")
            print(f"Segmenting point cloud: {point_cloud_path}")
            
            # Check if point cloud file exists and has points
            if not validate_point_cloud_file(point_cloud_path):
                sys.exit(1)
                
            segmenter = PointCloudSegmentation(max_planes=args.max_planes, 
                                             voxel_size=args.voxel_size,
                                             distance_threshold=args.distance_threshold)
            
            segmenter.load_and_preprocess(point_cloud_path)
            
            # Process point cloud and get result
            result = segmenter.segment_planes()
            if result is None:
                print("Error: Segmentation failed to produce results.")
                sys.exit(1)
                
            segmenter.refine_segments()
            
            segmentation_output_path = os.path.join(args.output_dir, 'segmented.ply')
            segmenter.save_result(segmentation_output_path)
            print(f"Segmentation completed. Result saved to: {segmentation_output_path}")
        
        except Exception as e:
            print(f"Error during segmentation: {e}")
    else:
        print("\nSkipping point cloud segmentation phase.")
    
    # Phase 3: Diffraction Edge Detection and Mesh Generation
    if not args.skip_diffraction:
        try:
            print("\n===== PHASE 3: DIFFRACTION EDGE DETECTION =====")
            
            # Use segmented point cloud if available, otherwise use original
            input_path = segmentation_output_path if segmentation_output_path else point_cloud_path
            print(f"Detecting diffraction edges from: {input_path}")
            
            # Validate the input point cloud
            if not validate_segmented_point_cloud(input_path):
                print("Error: Input point cloud is not properly segmented for diffraction edge detection")
                sys.exit(1)
            
            detector = DiffractionEdgeDetector()
            
            # Generate output path
            mesh_output_path = os.path.join(args.output_dir, 'diffraction_mesh.ply')
            
            # Process the point cloud and generate mesh with diffraction edges
            detector.process(
                input_file=input_path,
                output_file=mesh_output_path,
                visualize=False  # Set to True if visualization is desired
            )
            
            print(f"Diffraction edge detection completed. Results saved to: {mesh_output_path}")
            
        except Exception as e:
            print(f"Error during diffraction edge detection: {e}")
    else:
        print("\nSkipping diffraction edge detection phase.")
    
    print("\nPipeline completed successfully!")

# Helper functions for TSDF data setup
def setup_color_images(src_path, dst_path):
    """Setup color images in the expected directory structure."""
    # Implementation depends on your input data format
    pass

def setup_depth_images(src_path, dst_path):
    """Setup depth images in the expected directory structure."""
    # Implementation depends on your input data format
    pass

def setup_camera_poses(src_path, dst_file):
    """Setup camera poses file in the expected format."""
    # Implementation depends on your input data format
    pass

def setup_camera_calibration(src_path, dst_file):
    """Setup camera calibration file in the expected format."""
    # Implementation depends on your input data format
    pass

if __name__ == "__main__":
    main()