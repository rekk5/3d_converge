import os
import argparse
import sys
import numpy as np
import open3d as o3d

from colmap_recon import run_colmap
from tsdf_fusion import TSDFFusion
from segmentation import PointCloudSegmentation, validate_point_cloud_file
from diffraction import DiffractionEdgeDetector, validate_segmented_point_cloud
from diffraction_noseg import detect_edges_local_normals, save_ply_with_edges
from comparison import create_differential_pointcloud, create_differential_pointcloud_with_custom_ply, create_enhanced_differential_pointcloud_with_custom_ply
from vggt_recon import run_vggt
from naming_convention import get_naming_convention


def parse_arguments():
    parser = argparse.ArgumentParser(description='3D Reconstruction Pipeline')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, help='Path to the input images')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--project_name', type=str, default='reconstruction', help='Project name for consistent file naming')
    parser.add_argument('--recon_method', type=str, choices=['colmap', 'tsdf', 'mast3r', 'vggt', 'none'], default='vggt', 
                        help='Reconstruction method to use (vggt, mast3r, colmap, tsdf, or none to skip reconstruction)')
    parser.add_argument('--point_cloud_path', type=str, help='Path to existing point cloud (skip reconstruction)')
    
    # Comparison arguments
    parser.add_argument('--base_cloud_path', type=str, help='Path to base point cloud for comparison')
    parser.add_argument('--comparison_threshold', type=float, default=0.15, 
                        help='Distance threshold for difference detection (in meters)')
    parser.add_argument('--comparison_voxel_size', type=float, default=0.02, 
                        help='Voxel size for comparison downsampling')
    parser.add_argument('--selective_segmentation', action='store_true', 
                        help='Only segment areas that differ from the base cloud')
    parser.add_argument('--enhanced_comparison', action='store_true', 
                        help='Use enhanced comparison with multi-scale analysis, confidence scoring, and color coding')
    parser.add_argument('--use_multi_scale', action='store_true', default=True,
                        help='Use multi-scale analysis for better registration')
    parser.add_argument('--use_confidence', action='store_true', default=True,
                        help='Compute confidence scores for difference detection')
    parser.add_argument('--use_color_coding', action='store_true', default=True,
                        help='Use color-coded differences based on magnitude')
    
    # TSDF specific parameters
    parser.add_argument('--tsdf_data_path', type=str, help='Path to data directory with color, depth, poses, and calibration folders')
    parser.add_argument('--voxel_length', type=float, default=0.01, help='Voxel length for TSDF fusion')
    parser.add_argument('--sdf_trunc', type=float, default=0.04, help='Truncation distance for TSDF')
    
    # MAST3R specific parameters
    # VGGT specific parameters
    parser.add_argument('--conf_threshold', type=float, default=3.0, help='Confidence threshold for VGGT point filtering (default: 3.0)')
    parser.add_argument('--show_cameras', action='store_true', default=True, help='Show cameras in VGGT 3D scene (default: True)')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract from video for VGGT (default: 1.0)')
    parser.add_argument('--no_cameras', action='store_true', help='Do not show cameras in VGGT 3D scene')
    

    parser.add_argument('--image_size', type=int, default=512, help='Image size for MAST3R reconstruction (default: 512)')
    parser.add_argument('--resize_images', action='store_true', help='Resize images to 512x512 before MAST3R processing')
    
    # Point cloud segmentation parameters
    parser.add_argument('--skip_segmentation', action='store_true', help='Skip the segmentation phase')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size for downsampling')
    parser.add_argument('--max_planes', type=int, default=60, help='Maximum number of planes to detect')
    parser.add_argument('--distance_threshold', type=float, default=0.04, help='Distance threshold for plane detection')
    parser.add_argument('--skip_refinement', action='store_true', help='Skip the segment refinement/merging step')
    parser.add_argument('--enhanced_segmentation', action='store_true', help='Use enhanced RANSAC+DBSCAN segmentation method')
    
    # Diffraction edge parameters
    parser.add_argument('--skip_diffraction', action='store_true', help='Skip the diffraction edge detection phase')
    parser.add_argument('--diffraction_method', type=str, choices=['segmentation', 'normals'], default='segmentation',
                        help='Method for diffraction edge detection: segmentation-based or normal-based')
    parser.add_argument('--angle_threshold', type=float, default=45.0, help='Angle threshold for edge detection (degrees)')
    parser.add_argument('--min_edge_length', type=float, default=0.1, help='Minimum edge length to consider')
    # Normal-based edge detection parameters
    parser.add_argument('--normal_radius', type=float, default=0.05, help='Radius for normal estimation')
    parser.add_argument('--normal_max_nn', type=int, default=30, help='Max neighbors for normal estimation')
    parser.add_argument('--edge_radius', type=float, default=0.05, help='Radius for edge detection neighbor search')
    parser.add_argument('--voxel_downsampling', type=float, default=0.02, help='Voxel size for downsampling. Set to 0 to disable')
    parser.add_argument('--sor_neighbors', type=int, default=20, help='Neighbors for statistical outlier removal. Set to 0 to disable')
    parser.add_argument('--sor_std_ratio', type=float, default=2.0, help='Standard deviation ratio for outlier removal')
    # New: make mesh creation optional during diffraction
    parser.add_argument('--create_mesh', action='store_true', help='Create surface meshes during diffraction output (optional)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize naming convention
    naming = get_naming_convention(args.output_dir, args.project_name)
    naming.create_directories()
    
    print(f"=== 3D RECONSTRUCTION PIPELINE ===")
    print(f"Project: {args.project_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Organized output structure created with consistent naming")
    print()
    
    # Phase 1: 3D Reconstruction (optional)
    point_cloud_path = args.point_cloud_path
    
    if args.recon_method != 'none' and point_cloud_path is None:
        if args.recon_method == 'colmap':
            if args.image_path is None:
                print("Error: --image_path is required for COLMAP reconstruction")
                sys.exit(1)
                
            print("===== PHASE 1: COLMAP RECONSTRUCTION =====")
            print(f"Starting COLMAP reconstruction with images from: {args.image_path}")
            
            # Create output directory for COLMAP
            colmap_output_dir = os.path.join(naming.get_dir_path('reconstruction'), 'colmap')
            os.makedirs(colmap_output_dir, exist_ok=True)
            
            # Run COLMAP reconstruction
            point_cloud_path = run_colmap(
                image_path=args.image_path,
                database_path=os.path.join(colmap_output_dir, 'database.db'),
                sparse_output_path=os.path.join(colmap_output_dir, 'sparse'),
                dense_output_path=os.path.join(colmap_output_dir, 'dense')
            )
            
            # Move/rename the output to follow naming convention
            if point_cloud_path and os.path.exists(point_cloud_path):
                final_path = naming.file_patterns['colmap_point_cloud']
                os.rename(point_cloud_path, final_path)
                point_cloud_path = final_path
            
            print(f"COLMAP reconstruction completed. Point cloud saved to: {point_cloud_path}")
        
        elif args.recon_method == 'tsdf':
            print("===== PHASE 1: TSDF FUSION RECONSTRUCTION =====")
            
            # Validate required paths for TSDF
            if not args.tsdf_data_path:
                print("Error: TSDF fusion requires a data path with the expected folder structure.")
                print("Please provide --tsdf_data_path pointing to a directory containing:")
                print("  - color/ (RGB images)")
                print("  - depth_render/ (depth images)")
                print("  - poses/poses_color.txt (camera poses)")
                print("  - calibration/calib_color.yaml (camera calibration)")
                sys.exit(1)
            
            print(f"Starting TSDF fusion with data from: {args.tsdf_data_path}")
            
            # Check if the expected folder structure exists
            required_paths = [
                os.path.join(args.tsdf_data_path, 'color'),
                os.path.join(args.tsdf_data_path, 'depth_render'),
                os.path.join(args.tsdf_data_path, 'poses'),
                os.path.join(args.tsdf_data_path, 'calibration')
            ]
            
            for path in required_paths:
                if not os.path.exists(path):
                    print(f"Error: Required directory not found: {path}")
                    sys.exit(1)
            
            # Run TSDF fusion
            try:
                tsdf = TSDFFusion(
                    data_path=args.tsdf_data_path,
                    voxel_length=args.voxel_length,
                    sdf_trunc=args.sdf_trunc
                )
                
                pcd, mesh = tsdf.run(save_pcd=True, save_mesh=True)
                
                if pcd is not None:
                    # Get original path and move to naming convention path
                    original_path = os.path.join(args.tsdf_data_path, 'recon', 'point_cloud.ply')
                    point_cloud_path = naming.file_patterns['tsdf_point_cloud']
                    
                    if os.path.exists(original_path):
                        os.rename(original_path, point_cloud_path)
                    
                    # Handle mesh if it exists
                    original_mesh_path = os.path.join(args.tsdf_data_path, 'recon', 'mesh.ply')
                    if os.path.exists(original_mesh_path):
                        final_mesh_path = naming.file_patterns['tsdf_mesh']
                        os.rename(original_mesh_path, final_mesh_path)
                        print(f"TSDF mesh saved to: {final_mesh_path}")
                    
                    print(f"TSDF fusion completed. Point cloud saved to: {point_cloud_path}")
                else:
                    print("Error: TSDF fusion failed to generate a point cloud")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"Error during TSDF fusion: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        elif args.recon_method == 'mast3r':
            # Lazy import to avoid pulling heavy deps unless needed
            from mast3r_recon import run_mast3r
            print("===== PHASE 1: MAST3R RECONSTRUCTION =====")
            if args.image_path is None:
                print("Error: --image_path is required for MAST3R reconstruction")
                sys.exit(1)
            print(f"Starting MAST3R reconstruction with images from: {args.image_path}")
            mast3r_output_dir = os.path.join(naming.get_dir_path('reconstruction'), 'mast3r')
            os.makedirs(mast3r_output_dir, exist_ok=True)
            try:
                glb_path, ply_path = run_mast3r(
                    args.image_path,
                    mast3r_output_dir,
                    matching_conf_thr=1.0,
                    min_conf_thr=0.5,
                    as_pointcloud=True,
                    convert_to_ply=True,
                    image_size=args.image_size,
                    resize_images=args.resize_images
                )
                if ply_path and os.path.exists(ply_path):
                    # Move to naming convention path
                    final_path = naming.file_patterns['mast3r_point_cloud']
                    os.rename(ply_path, final_path)
                    point_cloud_path = final_path
                    print(f"MAST3R reconstruction completed. Point cloud saved to: {point_cloud_path}")
                else:
                    print("Error: MAST3R failed to produce a PLY point cloud.")
                    sys.exit(1)
            except Exception as e:
                print(f"Error during MAST3R reconstruction: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        elif args.recon_method == 'vggt':
            print("===== PHASE 1: VGGT RECONSTRUCTION =====")
            if args.image_path is None:
                print("Error: --image_path is required for VGGT reconstruction")
                sys.exit(1)
            print(f"Starting VGGT reconstruction with images from: {args.image_path}")
            vggt_output_dir = os.path.join(naming.get_dir_path('reconstruction'), 'vggt')
            os.makedirs(vggt_output_dir, exist_ok=True)
            try:
                glb_path, ply_path = run_vggt(
                    args.image_path,
                    vggt_output_dir,
                    conf_threshold=args.conf_threshold,
                    show_cameras=not args.no_cameras,
                    fps=args.fps,
                    convert_to_ply=True
                )
                if ply_path and os.path.exists(ply_path):
                    # Move to naming convention path
                    final_path = naming.file_patterns['vggt_point_cloud']
                    os.rename(ply_path, final_path)
                    point_cloud_path = final_path
                    print(f"VGGT reconstruction completed. Point cloud saved to: {point_cloud_path}")
                else:
                    print("Error: VGGT failed to produce a PLY point cloud.")
                    sys.exit(1)
            except Exception as e:
                print(f"Error during VGGT reconstruction: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    
    elif point_cloud_path is None:
        print("Error: You must either specify a reconstruction method or provide an existing point cloud path")
        sys.exit(1)
    else:
        print(f"Skipping reconstruction, using existing point cloud: {point_cloud_path}")
    
    # Point cloud comparison (optional)
    if args.base_cloud_path:
        print("\n===== POINT CLOUD COMPARISON =====")
        print(f"Comparing with base cloud: {args.base_cloud_path}")
        
        # Use the appropriate point cloud for comparison
        comparison_input = point_cloud_path if point_cloud_path else point_cloud_path
        
        # Create differential output path using naming convention
        differential_output_path = naming.file_patterns['differential_comparison']
        
        try:
            # Create differential point cloud with enhanced or basic comparison
            if args.enhanced_comparison:
                print("Using enhanced comparison with multi-scale analysis, confidence scoring, and color coding...")
                diff_cloud, differential_flags, comparison_stats = create_enhanced_differential_pointcloud_with_custom_ply(
                    args.base_cloud_path,
                    comparison_input,
                    differential_output_path,
                    threshold=args.comparison_threshold,
                    voxel_size=args.comparison_voxel_size,
                    use_multi_scale=args.use_multi_scale,
                    use_confidence=args.use_confidence,
                    use_color_coding=args.use_color_coding
                )
                print(f"Enhanced comparison statistics:")
                print(f"  Registration fitness: {comparison_stats['registration_fitness']:.4f}")
                print(f"  Registration RMSE: {comparison_stats['registration_rmse']:.4f}")
                print(f"  Percentage different: {comparison_stats['percentage_different']:.2f}%")
            else:
                print("Using basic comparison...")
                diff_cloud, differential_flags = create_differential_pointcloud_with_custom_ply(
                    args.base_cloud_path,
                    comparison_input,
                    differential_output_path,
                    threshold=args.comparison_threshold,
                    voxel_size=args.comparison_voxel_size
                )
            
            print(f"Differential comparison saved to: {differential_output_path}")
            
            # Update point cloud path for further processing if selective segmentation is enabled
            if args.selective_segmentation:
                point_cloud_path = differential_output_path
                print("Using differential point cloud for selective segmentation...")
                
        except Exception as e:
            print(f"Error during point cloud comparison: {e}")
            import traceback
            traceback.print_exc()
    
    # Phase 2: Point Cloud Segmentation (optional)
    segmentation_output_path = None
    if not args.skip_segmentation:
        try:
            print("\n===== PHASE 2: POINT CLOUD SEGMENTATION =====")
            print(f"Segmenting point cloud: {point_cloud_path}")
            
            # Check if point cloud file exists and has points
            if not validate_point_cloud_file(point_cloud_path):
                sys.exit(1)
                
            segmenter = PointCloudSegmentation(
                max_planes=args.max_planes, 
                voxel_size=args.voxel_size,
                distance_threshold=args.distance_threshold,
                enhanced_mode=args.enhanced_segmentation
            )
            
            segmenter.load_and_preprocess(point_cloud_path)
            
            # Process point cloud and get result
            result = segmenter.segment_planes()
            if result is None:
                print("Error: Segmentation failed to produce results.")
                sys.exit(1)
            
            # For selective segmentation, we're already working with only the differential points
            if args.selective_segmentation and args.base_cloud_path:
                segmentation_output_path = naming.file_patterns['segmented_differential']
                segmenter.save_result(segmentation_output_path)
                print(f"Segmented differential points saved to: {segmentation_output_path}")
                
                # Optionally, you can merge with the base cloud if needed for visualization
                # Uncomment the following lines if you want to combine with base cloud:
                # base_cloud = o3d.io.read_point_cloud(args.base_cloud_path)
                # segmented_changes = o3d.io.read_point_cloud(segmentation_output_path)
                # combined_cloud = base_cloud + segmented_changes
                # final_output_path = naming.get_file_path('segmentation', 'point_cloud', 'combined_with_base')
                # o3d.io.write_point_cloud(final_output_path, combined_cloud)
                # print(f"Combined with base cloud saved to: {final_output_path}")
            else:
                segmentation_output_path = naming.file_patterns['segmented_point_cloud']
                segmenter.save_result(segmentation_output_path)
            
            print(f"Segmentation completed. Result saved to: {segmentation_output_path}")
        
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping point cloud segmentation phase.")
    
    # Phase 3: Diffraction Edge Detection and Mesh Generation
    if not args.skip_diffraction:
        try:
            print("\n===== PHASE 3: DIFFRACTION EDGE DETECTION =====")
            
            # Use segmented point cloud if available, otherwise use original
            input_path = segmentation_output_path if segmentation_output_path else point_cloud_path
            print(f"Detecting diffraction edges from: {input_path}")
            
            # Generate output path using naming convention
            mesh_output_path = naming.file_patterns['diffraction_mesh']
            
            if args.diffraction_method == 'segmentation':
                # Validate the input point cloud for segmentation-based method
                if not validate_segmented_point_cloud(input_path):
                    print("Error: Input point cloud is not properly segmented for diffraction edge detection")
                    sys.exit(1)
                
                detector = DiffractionEdgeDetector(create_mesh=args.create_mesh)
                
                # Process the point cloud and generate mesh with diffraction edges
                detector.process(
                    input_file=input_path,
                    output_file=mesh_output_path,
                    visualize=False  # Set to True if visualization is desired
                )
                
                print(f"Segmentation-based diffraction edge detection completed. Results saved to: {mesh_output_path}")
            
            elif args.diffraction_method == 'normals':
                print(f"Using normal-based diffraction edge detection method")
                
                try:
                    # Load the point cloud
                    pcd = o3d.io.read_point_cloud(input_path)
                    if not pcd.has_points():
                        print(f"Error: Failed to load point cloud from {input_path} or it's empty.")
                        sys.exit(1)
                    
                    print(f"Successfully loaded point cloud with {len(pcd.points)} points.")
                    
                    # Preprocessing (voxel downsampling and outlier removal)
                    pcd_processed = pcd
                    
                    # 1. Voxel Downsampling (Optional)
                    if args.voxel_downsampling > 0:
                        print(f"Applying Voxel Downsampling with voxel size: {args.voxel_downsampling}")
                        pcd_downsampled = pcd_processed.voxel_down_sample(voxel_size=args.voxel_downsampling)
                        print(f"Point cloud downsampled from {len(pcd_processed.points)} to {len(pcd_downsampled.points)} points.")
                        pcd_processed = pcd_downsampled
                    
                    # 2. Statistical Outlier Removal (Optional)
                    if args.sor_neighbors > 0:
                        print(f"Applying Statistical Outlier Removal (nb_neighbors={args.sor_neighbors}, std_ratio={args.sor_std_ratio})")
                        pcd_processed_sor, ind = pcd_processed.remove_statistical_outlier(
                            nb_neighbors=args.sor_neighbors,
                            std_ratio=args.sor_std_ratio
                        )
                        num_removed = len(pcd_processed.points) - len(pcd_processed_sor.points)
                        print(f"Removed {num_removed} outlier points.")
                        pcd_processed = pcd_processed_sor
                    
                    # Detect edges using normal-based method
                    pcd_edges_colored, edge_points_coords = detect_edges_local_normals(
                        pcd_processed,
                        normal_radius=args.normal_radius,
                        normal_max_nn=args.normal_max_nn,
                        neighbor_radius=args.edge_radius,
                        angle_threshold_deg=args.angle_threshold
                    )
                    
                    # Save the results
                    save_ply_with_edges(mesh_output_path, pcd_edges_colored, edge_points_coords)
                    
                    print(f"Normal-based diffraction edge detection completed. Results saved to: {mesh_output_path}")
                    
                except Exception as e:
                    print(f"Error during normal-based edge detection: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
            
        except Exception as e:
            print(f"Error during diffraction edge detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping diffraction edge detection phase.")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
