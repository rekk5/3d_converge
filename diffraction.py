import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import argparse

def separate_planes_by_color(pcd, min_points=100):
    colors = np.asarray(pcd.colors)
    unique_colors = np.unique(colors, axis=0)
    planes = []
    
    print(f"\nFound {len(unique_colors)} unique colors in point cloud")
    print("Filtering planes...")
    
    for i, color in enumerate(unique_colors):
        color_mask = np.all(colors == color, axis=1)
        indices = np.where(color_mask)[0]
        num_points = len(indices)
        
        if num_points >= min_points:
            plane = pcd.select_by_index(indices)
            planes.append(plane)
            print(f"Plane {i}: {num_points} points - Kept")
        else:
            print(f"Plane {i}: {num_points} points - Filtered out (below threshold of {min_points})")
    
    print(f"\nKept {len(planes)} planes out of {len(unique_colors)} total\n")
    return planes

def fit_plane_ransac(points):
    if len(points) < 3:
        return None, None
    
    try:
        ransac = RANSACRegressor(min_samples=3, residual_threshold=0.02, max_trials=10000)
        X = points[:, :2]
        y = points[:, 2]
        ransac.fit(X, y)
        normal = np.array([-ransac.estimator_.coef_[0], -ransac.estimator_.coef_[1], 1])
        normal /= np.linalg.norm(normal)
        centroid = np.mean(points, axis=0)
        return normal, centroid
    except Exception as e:
        print(f"Error fitting plane: {e}")
        return None, None

def find_close_points(points1, points2, normal1, normal2, threshold=0.08):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    if angle < np.pi/6:
        return np.array([])
    
    close_points = []
    for point in points1:
        dist, _ = tree2.query(point)
        if dist < threshold:
            dist1 = np.abs(np.dot(point - points1[0], normal1))
            dist2 = np.abs(np.dot(point - points2[0], normal2))
            if dist1 < threshold and dist2 < threshold:
                close_points.append(point)
    
    return np.array(close_points)

def fit_line_to_points(points):
    if len(points) < 10:
        return None, None
    
    pca = PCA(n_components=1)
    pca.fit(points)
    
    if pca.explained_variance_ratio_[0] < 0.8:
        return None, None
    
    direction = pca.components_[0]
    centroid = np.mean(points, axis=0)
    projected_points = np.dot(points - centroid, direction[:, np.newaxis]) * direction + centroid
    distances = np.dot(projected_points - centroid, direction)
    min_idx, max_idx = np.argmin(distances), np.argmax(distances)
    
    edge_length = np.linalg.norm(projected_points[max_idx] - projected_points[min_idx])
    if edge_length < 0.5:
        return None, None
    
    return projected_points[min_idx], projected_points[max_idx]

def find_diffraction_edges(planes):
    diffraction_edges = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            points1 = np.asarray(planes[i].points)
            points2 = np.asarray(planes[j].points)
            
            if len(points1) < 3 or len(points2) < 3:
                continue
            
            plane_fit1 = fit_plane_ransac(points1)
            plane_fit2 = fit_plane_ransac(points2)
            
            if plane_fit1[0] is None or plane_fit2[0] is None:
                continue
                
            normal1, _ = plane_fit1
            normal2, _ = plane_fit2
            
            close_points = find_close_points(points1, points2, normal1, normal2)
            if len(close_points) > 1:
                start_point, end_point = fit_line_to_points(close_points)
                if start_point is not None and end_point is not None:
                    diffraction_edges.append((start_point, end_point))
    
    return diffraction_edges

def save_and_visualize_diffraction_edges(diffraction_edges, pcd, output_file):
    # Create a new point cloud for the edges
    edge_points = []
    edge_colors = []
    edge_lines = []
    point_index = 0
    
    # Add original point cloud points and colors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    edge_points.extend(points)
    edge_colors.extend(colors)
    base_index = len(points)
    
    # Add edge points and lines
    for start_point, end_point in diffraction_edges:
        edge_points.extend([start_point, end_point])
        edge_colors.extend([[1, 0, 0], [1, 0, 0]])  # Red color for edges
        edge_lines.append([base_index + point_index, base_index + point_index + 1])
        point_index += 2
    
    # Create new point cloud with edges
    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    edge_pcd.colors = o3d.utility.Vector3dVector(edge_colors)
    
    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(edge_points)
    line_set.lines = o3d.utility.Vector2iVector(edge_lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in edge_lines])  # Red lines
    
    # Save the combined point cloud
    o3d.io.write_point_cloud(output_file, edge_pcd)
    print(f"Combined point cloud saved to {output_file}")
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometry for visualization
    vis.add_geometry(edge_pcd)
    vis.add_geometry(line_set)
    
    # Improve rendering settings
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.line_width = 5.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def save_plane_info(planes, normals, output_base):
    """Save plane information to a text file."""
    info_file = output_base.rsplit('.', 1)[0] + '_planes.txt'
    with open(info_file, 'w') as f:
        f.write("Plane Information:\n")
        f.write("-----------------\n\n")
        for i, (plane, normal) in enumerate(zip(planes, normals)):
            points = np.asarray(plane.points)
            colors = np.asarray(plane.colors)[0]  # Get first color (all same)
            centroid = np.mean(points, axis=0)
            f.write(f"Plane {i}:\n")
            f.write(f"Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]\n")
            f.write(f"Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]\n")
            f.write(f"Color: [{colors[0]:.4f}, {colors[1]:.4f}, {colors[2]:.4f}]\n")
            f.write(f"Number of points: {len(points)}\n\n")

def main(input_file, output_file):
    # Load and visualize input point cloud
    pcd = o3d.io.read_point_cloud(input_file)
    print("\nInput point cloud:")
    print(f"Points: {len(pcd.points)}")
    print(f"Has normals: {pcd.has_normals()}")
    print(f"Has colors: {pcd.has_colors()}")
    
    # Initial visualization of input
    o3d.visualization.draw_geometries([pcd])
    
    # Process the point cloud
    planes = separate_planes_by_color(pcd, min_points=100)
    
    # Fit planes and store normals
    plane_normals = []
    for plane in planes:
        points = np.asarray(plane.points)
        normal, _ = fit_plane_ransac(points)
        if normal is not None:
            plane_normals.append(normal)
        else:
            plane_normals.append(np.array([0, 0, 1]))  # default if fit fails
    
    # Save plane information
    save_plane_info(planes, plane_normals, output_file)
    
    # Debug information about planes
    print(f"\nNumber of planes found: {len(planes)}")
    for i, plane in enumerate(planes):
        print(f"Plane {i}: {len(plane.points)} points")
    
    diffraction_edges = find_diffraction_edges(planes)
    
    if diffraction_edges:
        print(f"\nFound {len(diffraction_edges)} diffraction edges")
        save_and_visualize_diffraction_edges(diffraction_edges, pcd, output_file)
        print(f"Diffraction edges saved to {output_file}")
    else:
        print("\nNo diffraction edges found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a segmented point cloud to find diffraction edges.")
    parser.add_argument("input_file", type=str, help="Path to the input segmented point cloud .ply file")
    parser.add_argument("output_file", type=str, help="Path to save the output .ply file with diffraction edges")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)