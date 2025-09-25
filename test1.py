import open3d as o3d
import numpy as np

def add_sphere_to_pointcloud(point_cloud, center, radius, density=100, color=[1, 0, 0]):
    """
    Adds a sphere to an existing point cloud
    
    Args:
        point_cloud: o3d.geometry.PointCloud object or path to point cloud
        center: List/tuple of [x, y, z] coordinates for sphere center
        radius: Radius of the sphere
        density: Number of points to generate along each dimension (default: 100)
        color: RGB color for the sphere points (default: red)
    
    Returns:
        Modified point cloud with added sphere
    """
    # Load point cloud if path is provided
    if isinstance(point_cloud, str):
        cloud = o3d.io.read_point_cloud(point_cloud)
    else:
        cloud = o3d.geometry.PointCloud(point_cloud)  # Create a new point cloud instance
    
    # Create sphere points using spherical coordinates
    phi = np.linspace(0, 2*np.pi, density)
    theta = np.linspace(0, np.pi, density)
    phi, theta = np.meshgrid(phi, theta)
    
    # Convert to Cartesian coordinates
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    
    # Reshape to Nx3 array
    sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    # Create colors array for sphere points
    sphere_colors = np.tile(color, (len(sphere_points), 1))
    
    # Combine original and sphere points/colors
    combined_points = np.vstack([np.asarray(cloud.points), sphere_points])
    
    # Create new point cloud with combined points
    modified_cloud = o3d.geometry.PointCloud()
    modified_cloud.points = o3d.utility.Vector3dVector(combined_points)
    
    if cloud.has_colors():
        # Keep original colors and add sphere colors
        combined_colors = np.vstack([np.asarray(cloud.colors), sphere_colors])
        modified_cloud.colors = o3d.utility.Vector3dVector(combined_colors)
    else:
        # If no colors in original cloud, only add colors for the sphere points
        colors = np.zeros((len(combined_points), 3))  # Initialize all colors as None
        colors[-len(sphere_points):] = sphere_colors  # Only set colors for sphere points
        modified_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove colors from the original points by setting them to None
        colors_array = np.asarray(modified_cloud.colors)
        colors_array[:len(cloud.points)] = None
    
    return modified_cloud

# Example usage
if __name__ == "__main__":
    # Load original point cloud
    cloud = o3d.io.read_point_cloud("/home/t/Desktop/insectec-seg.ply")
    
    # Create modified version with a sphere
    modified_cloud = add_sphere_to_pointcloud(
        cloud,
        center=[0, 0, 0],  # Adjust center coordinates as needed
        radius=0.1,        # Adjust radius as needed
        density=100,
        color=[1, 0, 0]    # Red color
    )
    
    # Save modified cloud
    o3d.io.write_point_cloud("/home/t/Desktop/insectec_modified.ply", modified_cloud)