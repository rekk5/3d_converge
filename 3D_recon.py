import os
import argparse
import pycolmap
from pathlib import Path
import numpy as np

def run_reconstruction(image_dir, output_dir, gpu=True):
    """
    Run the COLMAP reconstruction pipeline using pycolmap
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to store COLMAP output
        gpu: Whether to use GPU acceleration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up paths
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Configure options
    sift_options = pycolmap.SiftExtractionOptions()
    # Set GPU usage for feature extraction
    if gpu:
        sift_options.gpu_index = "0"  # Use first GPU
    else:
        sift_options.gpu_index = "-1"  # Use CPU
    
    reader_options = pycolmap.ImageReaderOptions()
    
    # Run feature extraction
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        camera_model="SIMPLE_RADIAL",
        reader_options=reader_options,
        sift_options=sift_options
    )
    
    # Configure matching options
    sift_matching_options = pycolmap.SiftMatchingOptions()
    if gpu:
        sift_matching_options.gpu_index = "0"  # Use first GPU
    else:
        sift_matching_options.gpu_index = "-1"  # Use CPU
    
    matching_options = pycolmap.ExhaustiveMatchingOptions()
    verification_options = pycolmap.TwoViewGeometryOptions()
    
    # Run matching with correct parameters
    pycolmap.match_exhaustive(
        database_path,
        sift_matching_options,
        matching_options,
        verification_options
    )
    
    # Run mapper (SfM)
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sparse_dir,
        pipeline_options
    )
    
    # Get the first reconstruction (if any were created)
    reconstruction = None
    if reconstructions and len(reconstructions) > 0:
        reconstruction = list(reconstructions.values())[0]
    
    return sparse_dir, reconstruction

def run_dense_reconstruction(sparse_dir, image_dir, output_dir, reconstruction):
    """
    Run the COLMAP dense reconstruction pipeline
    
    Args:
        sparse_dir: Directory containing sparse reconstruction
        image_dir: Directory containing input images
        output_dir: Directory to store dense reconstruction
        reconstruction: Reconstruction object from sparse reconstruction
    """
    # Create dense output directory
    dense_dir = os.path.join(output_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)
    
    # Get the specific reconstruction folder (the first one, numbered "0")
    sparse_model_path = os.path.join(sparse_dir, "0")
    
    # Check if the sparse model exists
    if not os.path.exists(sparse_model_path):
        print(f"Error: Sparse reconstruction not found at {sparse_model_path}")
        print(f"Contents of {sparse_dir}:")
        print(os.listdir(sparse_dir))
        return None
    
    try:
        # The correct order is: output_path, input_path (sparse model path), image_path
        print("Running undistort_images...")
        pycolmap.undistort_images(
            dense_dir,          # output_path
            sparse_model_path,  # input_path - this is where the cameras.bin, images.bin files are
            image_dir           # image_path
        )
        
        # Check if patch_match_stereo is available
        if hasattr(pycolmap, 'patch_match_stereo'):
            print("Running patch_match_stereo...")
            pycolmap.patch_match_stereo(
                dense_dir,  # workspace_path
                workspace_format='COLMAP',
                pmvs_option_name='option-all'
            )
        else:
            print("Warning: pycolmap.patch_match_stereo function not available in this version of pycolmap.")
            print("You may need to run this step manually with the COLMAP command line tool:")
            print(f"colmap patch_match_stereo --workspace_path {dense_dir} --workspace_format COLMAP")
            
        # Check if stereo_fusion is available
        if hasattr(pycolmap, 'stereo_fusion'):
            print("Running stereo_fusion...")
            ply_path = os.path.join(dense_dir, "fused.ply")
            
            # Based on the error message, the correct parameters for stereo_fusion are:
            # output_path, workspace_path, workspace_format, pmvs_option_name, input_type, options
            fusion_options = pycolmap.StereoFusionOptions()
            pycolmap.stereo_fusion(
                ply_path,       # output_path
                dense_dir,      # workspace_path
                'COLMAP',       # workspace_format
                'option-all',   # pmvs_option_name
                'geometric',    # input_type
                fusion_options  # options
            )
        else:
            print("Warning: pycolmap.stereo_fusion function not available in this version of pycolmap.")
            print("You may need to run this step manually with the COLMAP command line tool:")
            print(f"colmap stereo_fusion --workspace_path {dense_dir} --output_path {os.path.join(dense_dir, 'fused.ply')}")
    
    except Exception as e:
        print(f"Error during dense reconstruction: {str(e)}")
        print("Dense reconstruction failed, but sparse reconstruction is still available.")
    
    return dense_dir

def main():
    parser = argparse.ArgumentParser(description="3D reconstruction using pycolmap")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to store COLMAP output")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()
    
    # Run structure from motion
    sparse_dir, reconstruction = run_reconstruction(args.image_dir, args.output_dir, not args.no_gpu)
    print(f"Sparse reconstruction completed. Results saved in {sparse_dir}")
    
    # Run dense reconstruction
    dense_dir = run_dense_reconstruction(sparse_dir, args.image_dir, args.output_dir, reconstruction)
    print(f"Dense reconstruction completed. Results saved in {dense_dir}")
    print(f"Fused point cloud saved as {os.path.join(dense_dir, 'fused.ply')}")

if __name__ == "__main__":
    main()