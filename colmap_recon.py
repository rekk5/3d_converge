import subprocess
import os

def run_colmap(image_path="rgb", 
               database_path="database.db", 
               sparse_output_path="sparse",
               dense_output_path="dense"):
    """
    Run the COLMAP reconstruction pipeline with the exact parameters
    that produced good results previously.
    """
    # Create output directories if they don't exist
    os.makedirs(sparse_output_path, exist_ok=True)
    os.makedirs(dense_output_path, exist_ok=True)
    
    # Verify that the image directory exists and contains images
    if not os.path.exists(image_path):
        raise ValueError(f"Image directory '{image_path}' does not exist")
    
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError(f"No image files found in '{image_path}'")
    
    print(f"Found {len(image_files)} images in {image_path}")
    
    # Step 1: Feature extraction
    print("Running feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.max_image_size", "1440",
        "--SiftExtraction.peak_threshold", "0.009",
        "--SiftExtraction.edge_threshold", "6",
        "--SiftExtraction.max_num_features", "20000"
    ], check=True)
    
    # Step 2: Feature matching
    print("Running feature matching...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "65536",
        "--SiftMatching.max_ratio", "0.8",
        "--TwoViewGeometry.max_error", "5",
        "--TwoViewGeometry.min_num_inliers", "16",
        "--TwoViewGeometry.min_inlier_ratio", "0.125"
    ], check=True)
    
    # Step 3: Sparse reconstruction
    print("Running sparse reconstruction...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sparse_output_path,
        "--Mapper.abs_pose_min_num_inliers", "16",
        "--Mapper.abs_pose_min_inlier_ratio", "0.15",
        "--Mapper.filter_max_reproj_error", "5",
        "--Mapper.init_min_tri_angle", "1.75",
        "--Mapper.ba_global_max_num_iterations", "200"
    ], check=True)
    
    # Step 4: Image undistortion
    print("Running image undistortion...")
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", image_path,
        "--input_path", f"{sparse_output_path}/0",
        "--output_path", dense_output_path,
        "--output_type", "COLMAP",
        "--max_image_size", "1440"
    ], check=True)
    
    # Step 5: Dense stereo matching
    print("Running dense stereo matching...")
    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", dense_output_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.max_image_size", "1440",
        "--PatchMatchStereo.window_radius", "6",
        "--PatchMatchStereo.window_step", "2",
        "--PatchMatchStereo.num_iterations", "5",
        "--PatchMatchStereo.filter_min_ncc", "0.075",
        "--PatchMatchStereo.filter_min_triangulation_angle", "0.8",
        "--PatchMatchStereo.filter_min_num_consistent", "2",
        "--PatchMatchStereo.filter_geom_consistency_max_cost", "1.0"
    ], check=True)
    
    # Step 6: Dense stereo fusion
    print("Running dense stereo fusion...")
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", dense_output_path,
        "--workspace_format", "COLMAP",
        "--output_path", f"{dense_output_path}/fused.ply",
        "--StereoFusion.min_num_pixels", "2",
        "--StereoFusion.max_reproj_error", "2",
        "--StereoFusion.max_depth_error", "0.05"
    ], check=True)

    # Verify the output point cloud exists and has content
    if os.path.exists(f"{dense_output_path}/fused.ply"):
        if os.path.getsize(f"{dense_output_path}/fused.ply") > 100:  # Check if file has content
            print("COLMAP reconstruction pipeline completed successfully!")
            return f"{dense_output_path}/fused.ply"
        else:
            print("Warning: Output point cloud file is empty or very small")
    else:
        print("Error: Output point cloud file was not created")
    
    return f"{dense_output_path}/fused.ply"