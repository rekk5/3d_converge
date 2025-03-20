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
    
    # Step 1: Feature extraction
    print("Running feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.max_image_size", "1440",
        "--SiftExtraction.peak_threshold", "0.01",
        "--SiftExtraction.edge_threshold", "5",
        "--SiftExtraction.max_num_features", "32768"
    ], check=True)
    
    # Step 2: Feature matching
    print("Running feature matching...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "65536",
        "--SiftMatching.max_ratio", "0.6",
        "--TwoViewGeometry.max_error", "2",
        "--TwoViewGeometry.min_num_inliers", "20",
        "--TwoViewGeometry.min_inlier_ratio", "0.2"
    ], check=True)
    
    # Step 3: Sparse reconstruction
    print("Running sparse reconstruction...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sparse_output_path,
        "--Mapper.abs_pose_min_num_inliers", "20",
        "--Mapper.abs_pose_min_inlier_ratio", "0.2",
        "--Mapper.filter_max_reproj_error", "2",
        "--Mapper.init_min_tri_angle", "3",
        "--Mapper.ba_global_max_num_iterations", "400"
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
        "--PatchMatchStereo.window_radius", "7",
        "--PatchMatchStereo.window_step", "1",
        "--PatchMatchStereo.num_iterations", "5",
        "--PatchMatchStereo.filter_min_ncc", "0.2",
        "--PatchMatchStereo.filter_min_triangulation_angle", "2.0",
        "--PatchMatchStereo.filter_min_num_consistent", "2",
        "--PatchMatchStereo.filter_geom_consistency_max_cost", "0.5"
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
        "--StereoFusion.max_depth_error", "0.005"
    ], check=True)

    print("COLMAP reconstruction pipeline completed successfully!")
    return f"{dense_output_path}/fused.ply"