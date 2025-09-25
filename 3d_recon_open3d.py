import open3d as o3d
import numpy as np
import glob
import os

# Update paths to your actual RGB and depth images
rgb_folder = "/home/t/Desktop/recon/rgb"
depth_folder = "/home/t/Desktop/recon/depth"

rgb_paths = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
depth_paths = sorted(glob.glob(os.path.join(depth_folder, "*.png")))

# Check data consistency
assert len(rgb_paths) == len(depth_paths), "RGB and Depth folders must have the same number of images."

# Camera intrinsic parameters (adjust to match your actual sensor)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=1440, height=1056, fx=1400, fy=1400, cx=720, cy=528
)

# TSDF volume for integration
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.005,  # Adjust voxel resolution if needed
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# Initial camera pose
current_pose = np.eye(4)

# Load first RGB-D image to initialize
color_raw_prev = o3d.io.read_image(rgb_paths[0])
depth_raw_prev = o3d.io.read_image(depth_paths[0])

rgbd_prev = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw_prev, depth_raw_prev, depth_scale=1000.0, convert_rgb_to_intensity=False
)

# Integrate first frame
volume.integrate(rgbd_prev, intrinsic, np.linalg.inv(current_pose))

# Loop through all remaining frames
for i in range(1, len(rgb_paths)):
    color_raw = o3d.io.read_image(rgb_paths[i])
    depth_raw = o3d.io.read_image(depth_paths[i])

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, convert_rgb_to_intensity=False
    )

    # Odometry to estimate relative pose
    option = o3d.pipelines.odometry.OdometryOption()
    success, odo_transform, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_prev, rgbd, intrinsic, np.eye(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option
    )

    if success:
        # Update current global pose
        current_pose = current_pose @ np.linalg.inv(odo_transform)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(current_pose))
        print(f"Integrated frame {i+1}/{len(rgb_paths)} successfully.")
    else:
        print(f"Odometry failed at frame {i+1}, skipping integration for this frame.")

    # Update previous frame
    rgbd_prev = rgbd

# Extract mesh and compute normals
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

# Save mesh to disk
output_mesh_path = "/home/t/Desktop/recon/reconstructed_room.ply"
o3d.io.write_triangle_mesh(output_mesh_path, mesh)
print(f"Saved reconstructed mesh to {output_mesh_path}")

# Visualize the reconstructed mesh
o3d.visualization.draw_geometries([mesh])


