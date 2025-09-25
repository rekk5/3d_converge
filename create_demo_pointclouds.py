import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from comparison import create_differential_pointcloud_with_custom_ply

def create_base_scene():
    """Create a base scene with geometric objects"""
    
    # Create a room-like environment
    points = []
    colors = []
    
    # Floor (blue-ish)
    floor_x, floor_z = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    floor_y = np.zeros_like(floor_x) - 2
    floor_points = np.column_stack([floor_x.ravel(), floor_y.ravel(), floor_z.ravel()])
    floor_colors = np.tile([0.3, 0.3, 0.7], (len(floor_points), 1))  # Blue floor
    
    points.append(floor_points)
    colors.append(floor_colors)
    
    # Walls
    # Back wall (gray)
    wall_x, wall_y = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(-2, 3, 40))
    wall_z = np.ones_like(wall_x) * 5
    wall_points = np.column_stack([wall_x.ravel(), wall_y.ravel(), wall_z.ravel()])
    wall_colors = np.tile([0.6, 0.6, 0.6], (len(wall_points), 1))  # Gray wall
    
    points.append(wall_points)
    colors.append(wall_colors)
    
    # Add some furniture-like objects
    # Table (brown)
    table_center = np.array([0, -1, 0])
    table_size = np.array([2, 0.1, 1])
    table_x, table_y, table_z = np.meshgrid(
        np.linspace(-table_size[0]/2, table_size[0]/2, 20),
        np.linspace(-table_size[1]/2, table_size[1]/2, 3),
        np.linspace(-table_size[2]/2, table_size[2]/2, 10)
    )
    table_points = np.column_stack([table_x.ravel(), table_y.ravel(), table_z.ravel()]) + table_center
    table_colors = np.tile([0.6, 0.3, 0.1], (len(table_points), 1))  # Brown table
    
    points.append(table_points)
    colors.append(table_colors)
    
    # Chairs (darker brown)
    for chair_pos in [[-1.5, -1.5, 0.5], [1.5, -1.5, 0.5], [0, -1.5, -1.2]]:
        chair_x, chair_y, chair_z = np.meshgrid(
            np.linspace(-0.3, 0.3, 8),
            np.linspace(-0.3, 0.5, 10),
            np.linspace(-0.3, 0.3, 8)
        )
        chair_points = np.column_stack([chair_x.ravel(), chair_y.ravel(), chair_z.ravel()]) + np.array(chair_pos)
        chair_colors = np.tile([0.4, 0.2, 0.05], (len(chair_points), 1))  # Dark brown chairs
        
        points.append(chair_points)
        colors.append(chair_colors)
    
    # Combine all points
    all_points = np.vstack(points)
    all_colors = np.vstack(colors)
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02, all_points.shape)
    all_points += noise
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    pcd.estimate_normals()
    
    return pcd

def create_modified_scene():
    """Create a modified version of the base scene with new objects and changes"""
    
    # Start with the base scene
    base_pcd = create_base_scene()
    base_points = np.asarray(base_pcd.points)
    base_colors = np.asarray(base_pcd.colors)
    
    # Add new objects to create differences
    new_points = []
    new_colors = []
    
    # Add a laptop on the table (silver/black)
    laptop_center = np.array([0.5, -0.8, 0])
    laptop_x, laptop_y, laptop_z = np.meshgrid(
        np.linspace(-0.4, 0.4, 15),
        np.linspace(-0.02, 0.02, 2),
        np.linspace(-0.3, 0.3, 10)
    )
    laptop_points = np.column_stack([laptop_x.ravel(), laptop_y.ravel(), laptop_z.ravel()]) + laptop_center
    laptop_colors = np.tile([0.1, 0.1, 0.1], (len(laptop_points), 1))  # Black laptop
    
    new_points.append(laptop_points)
    new_colors.append(laptop_colors)
    
    # Add a coffee cup (white)
    cup_center = np.array([-0.8, -0.7, 0])
    cup_theta = np.linspace(0, 2*np.pi, 20)
    cup_height = np.linspace(0, 0.3, 15)
    cup_r = 0.08
    
    cup_points_list = []
    for h in cup_height:
        for theta in cup_theta:
            x = cup_r * np.cos(theta) + cup_center[0]
            y = h + cup_center[1]
            z = cup_r * np.sin(theta) + cup_center[2]
            cup_points_list.append([x, y, z])
    
    cup_points = np.array(cup_points_list)
    cup_colors = np.tile([0.9, 0.9, 0.9], (len(cup_points), 1))  # White cup
    
    new_points.append(cup_points)
    new_colors.append(cup_colors)
    
    # Add some books stacked (various colors)
    book_colors_list = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]]  # Red, Green, Blue books
    for i, book_color in enumerate(book_colors_list):
        book_center = np.array([1.2, -0.75 + i*0.05, 0.3])
        book_x, book_y, book_z = np.meshgrid(
            np.linspace(-0.15, 0.15, 8),
            np.linspace(-0.02, 0.02, 2),
            np.linspace(-0.2, 0.2, 10)
        )
        book_points = np.column_stack([book_x.ravel(), book_y.ravel(), book_z.ravel()]) + book_center
        book_colors_array = np.tile(book_color, (len(book_points), 1))
        
        new_points.append(book_points)
        new_colors.append(book_colors_array)
    
    # Add a potted plant (green)
    plant_center = np.array([2, -0.6, 2])
    
    # Pot (brown)
    pot_theta = np.linspace(0, 2*np.pi, 15)
    pot_height = np.linspace(0, 0.3, 10)
    pot_r = 0.15
    
    pot_points_list = []
    for h in pot_height:
        for theta in pot_theta:
            x = pot_r * np.cos(theta) + plant_center[0]
            y = h + plant_center[1]
            z = pot_r * np.sin(theta) + plant_center[2]
            pot_points_list.append([x, y, z])
    
    pot_points = np.array(pot_points_list)
    pot_colors = np.tile([0.4, 0.2, 0.1], (len(pot_points), 1))  # Brown pot
    
    new_points.append(pot_points)
    new_colors.append(pot_colors)
    
    # Plant leaves (green, random positions above pot)
    np.random.seed(42)  # For reproducible results
    num_leaves = 200
    leaf_points = []
    for _ in range(num_leaves):
        # Random position around and above the pot
        r = np.random.uniform(0, 0.2)
        theta = np.random.uniform(0, 2*np.pi)
        height = np.random.uniform(0.3, 0.8)
        
        x = r * np.cos(theta) + plant_center[0]
        y = height + plant_center[1]
        z = r * np.sin(theta) + plant_center[2]
        leaf_points.append([x, y, z])
    
    leaf_points = np.array(leaf_points)
    leaf_colors = np.tile([0.2, 0.8, 0.2], (len(leaf_points), 1))  # Green leaves
    
    new_points.append(leaf_points)
    new_colors.append(leaf_colors)
    
    # Combine original scene with new objects
    all_new_points = np.vstack(new_points)
    all_new_colors = np.vstack(new_colors)
    
    combined_points = np.vstack([base_points, all_new_points])
    combined_colors = np.vstack([base_colors, all_new_colors])
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02, all_new_points.shape)
    all_new_points += noise
    
    # Create modified point cloud
    modified_pcd = o3d.geometry.PointCloud()
    modified_pcd.points = o3d.utility.Vector3dVector(combined_points)
    modified_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    modified_pcd.estimate_normals()
    
    return modified_pcd

def save_demo_clouds():
    """Generate and save the demo point clouds"""
    
    print("Creating base scene...")
    base_cloud = create_base_scene()
    base_path = "demo_base_scene.ply"
    o3d.io.write_point_cloud(base_path, base_cloud)
    print(f"Base scene saved to: {base_path}")
    print(f"Base scene points: {len(base_cloud.points)}")
    
    print("\nCreating modified scene...")
    modified_cloud = create_modified_scene()
    modified_path = "demo_modified_scene.ply"
    o3d.io.write_point_cloud(modified_path, modified_cloud)
    print(f"Modified scene saved to: {modified_path}")
    print(f"Modified scene points: {len(modified_cloud.points)}")
    
    return base_path, modified_path

def run_differential_demo():
    """Create demo point clouds and run differential analysis"""
    
    print("=== Creating Demo Point Clouds for Differential Analysis ===\n")
    
    # Generate the demo point clouds
    base_path, modified_path = save_demo_clouds()
    
    print("\n=== Running Differential Analysis ===")
    
    # Run the differential comparison
    output_path = "demo_differential_result.ply"
    diff_cloud, differential_flags = create_differential_pointcloud_with_custom_ply(
        base_path,
        modified_path,
        output_path,
        threshold=0.1,  # Adjust threshold for demo
        voxel_size=0.05  # Slightly larger voxel for faster processing
    )
    
    print(f"\n=== Demo Complete ===")
    print(f"Files created:")
    print(f"  - Base scene: {base_path}")
    print(f"  - Modified scene: {modified_path}")
    print(f"  - Differential result: {output_path}")
    print(f"\nYou can visualize these files with Open3D or your preferred point cloud viewer!")
    
    return base_path, modified_path, output_path

if __name__ == "__main__":
    run_differential_demo() 