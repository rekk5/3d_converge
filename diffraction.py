import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import argparse
import struct
from plyfile import PlyData, PlyElement

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

def find_close_points(points1, points2, normal1, normal2, threshold=0.2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Calculate angle between normals
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    # Adjust angle thresholds for better detection
    if angle < np.pi/6 or angle > 5*np.pi/6:  # Skip if planes are too parallel or perpendicular
        return np.array([])
    
    close_points = []
    # Use vectorized operations for better performance
    distances, indices = tree2.query(points1, k=1)
    mask = distances < threshold
    
    for point, dist, idx in zip(points1[mask], distances[mask], indices[mask]):
        # Calculate distances to both planes
        dist1 = np.abs(np.dot(point - points1[0], normal1))
        dist2 = np.abs(np.dot(point - points2[0], normal2))
        
        # Add additional checks for point distribution
        if dist1 < threshold and dist2 < threshold:
            close_points.append(point)
    
    return np.array(close_points)

def fit_line_to_points(points):
    if len(points) < 55:  # Increased minimum points threshold
        return None, None
    
    pca = PCA(n_components=2)  # Use 2 components to check planarity
    pca.fit(points)
    
    # Check if points form a line (first component should dominate)
    if pca.explained_variance_ratio_[0] < 0.95:  # Increased threshold
        return None, None
    
    # Check if points are not too spread in second component
    if pca.explained_variance_ratio_[1] > 0.05:
        return None, None
    
    direction = pca.components_[0]
    centroid = np.mean(points, axis=0)
    
    # Project points onto the line
    projected_points = np.dot(points - centroid, direction[:, np.newaxis]) * direction + centroid
    distances = np.dot(projected_points - centroid, direction)
    
    # Find the endpoints
    min_idx, max_idx = np.argmin(distances), np.argmax(distances)
    
    # Check edge length and point distribution
    edge_length = np.linalg.norm(projected_points[max_idx] - projected_points[min_idx])
    if edge_length < 0.5 or edge_length > 25.0:  # Add maximum length threshold
        return None, None
    
    # Check point density along the line
    sorted_distances = np.sort(distances)
    gaps = sorted_distances[1:] - sorted_distances[:-1]
    if np.max(gaps) > edge_length * 0.5:  # Check for large gaps
        return None, None
    
    return projected_points[min_idx], projected_points[max_idx]

def find_diffraction_edges(planes):
    diffraction_edges = []
    edge_scores = []
    plane_pairs = []  # Store which planes form each edge
    
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
                
            normal1, centroid1 = plane_fit1
            normal2, centroid2 = plane_fit2
            
            # Check distance between plane centroids
            centroid_dist = np.linalg.norm(centroid1 - centroid2)
            if centroid_dist > 5.0:  # Skip if planes are too far apart
                continue
            
            close_points = find_close_points(points1, points2, normal1, normal2)
            if len(close_points) > 1:
                start_point, end_point = fit_line_to_points(close_points)
                if start_point is not None and end_point is not None:
                    edge_length = np.linalg.norm(end_point - start_point)
                    point_density = len(close_points) / edge_length
                    score = point_density * edge_length
                    
                    diffraction_edges.append(((start_point, end_point), (i, j)))
                    edge_scores.append(score)
    
    # Filter edges based on scores
    if edge_scores:
        median_score = np.median(edge_scores)
        filtered_edges = [edge for edge, score in zip(diffraction_edges, edge_scores) 
                        if score > median_score * 0.5]
        return filtered_edges
    
    return diffraction_edges

def save_and_visualize_diffraction_edges(diffraction_edges, pcd, output_file):
    # Create a new point cloud with just the original points
    edge_pcd = o3d.geometry.PointCloud()
    edge_pcd.points = pcd.points
    edge_pcd.colors = pcd.colors
    
    # Create line set for diffraction edges
    edge_lines = []
    line_points = []
    
    # Add all edge lines
    for (start_point, end_point), (plane1_idx, plane2_idx) in diffraction_edges:
        line_points.extend([start_point, end_point])
        edge_lines.append([len(line_points)-2, len(line_points)-1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(edge_lines)
    line_set.paint_uniform_color([1, 0, 0])  # Bright red
    
    # Save the point cloud
    o3d.io.write_point_cloud(output_file, edge_pcd)
    print(f"Point cloud saved to {output_file}")
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    vis.add_geometry(edge_pcd)
    vis.add_geometry(line_set)
    
    # Improve rendering settings
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.line_width = 5.0  # Make lines thicker
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def save_comprehensive_ply(points, colors, normals, labels, diffraction_edges, output_file):
    # Convert colors from float [0,1] to uint8 [0,255] if needed
    if colors.max() <= 1.0:
        colors = (colors * 255).astype('uint8')
    
    # Create structured array for vertex data
    vertex_data = []
    for p, c, n, l in zip(points, colors, normals, labels):
        vertex_data.append((
            float(p[0]), float(p[1]), float(p[2]),    # x, y, z (as float64)
            int(c[0]), int(c[1]), int(c[2]),          # red, green, blue (as uint8)
            float(n[0]), float(n[1]), float(n[2]),    # nx, ny, nz (as float32)
            int(l),                                    # label (as uint32)
            0                                          # material (as uint32)
        ))
    
    # Define vertex element with correct data types
    vertex = np.array(vertex_data,
        dtype=[
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),           # positions as float64
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # colors as uint8
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),        # normals as float32
            ('label', 'u4'), ('material', 'u4')              # indices as uint32
        ])
    
    # Create edge data
    edge_data = []
    for (start_point, end_point), (plane1_idx, plane2_idx) in diffraction_edges:
        edge_data.append((
            float(start_point[0]), float(start_point[1]), float(start_point[2]),
            float(end_point[0]), float(end_point[1]), float(end_point[2]),
            int(plane1_idx),
            int(plane2_idx)
        ))
    
    # Define edge element if we have edges
    if edge_data:
        edge = np.array(edge_data,
            dtype=[
                ('start_x', 'f8'), ('start_y', 'f8'), ('start_z', 'f8'),
                ('end_x', 'f8'), ('end_y', 'f8'), ('end_z', 'f8'),
                ('plane1', 'u4'), ('plane2', 'u4')
            ])
        
        # Create PLY file with both vertices and edges
        vertex_element = PlyElement.describe(vertex, 'vertex')
        edge_element = PlyElement.describe(edge, 'edge')
        PlyData([vertex_element, edge_element], text=True).write(output_file)
    else:
        # Create PLY file with only vertices
        vertex_element = PlyElement.describe(vertex, 'vertex')
        PlyData([vertex_element], text=True).write(output_file)

def main(input_file, output_file):
    # Load and visualize input point cloud
    pcd = o3d.io.read_point_cloud(input_file)
    print("\nInput point cloud:")
    print(f"Points: {len(pcd.points)}")
    print(f"Has normals: {pcd.has_normals()}")
    print(f"Has colors: {pcd.has_colors()}")
    
    # Initial visualization of input
    print("\nShowing input point cloud. Close window to continue...")
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
    
    # Collect all point cloud data
    all_points = []
    all_colors = []
    all_normals = []
    all_labels = []
    
    for i, (plane, normal) in enumerate(zip(planes, plane_normals)):
        points = np.asarray(plane.points)
        colors = np.asarray(plane.colors)
        normals = np.tile(normal, (len(points), 1))  # repeat normal for each point
        labels = np.full(len(points), i)  # assign plane index as label
        
        all_points.extend(points)
        all_colors.extend(colors)
        all_normals.extend(normals)
        all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_points = np.array(all_points)
    all_colors = np.array(all_colors)
    all_normals = np.array(all_normals)
    all_labels = np.array(all_labels)
    
    # Find diffraction edges and save everything to PLY
    diffraction_edges = find_diffraction_edges(planes)
    
    if diffraction_edges:
        print(f"\nFound {len(diffraction_edges)} diffraction edges")
        save_comprehensive_ply(all_points, all_colors, all_normals, 
                             all_labels, diffraction_edges, output_file)
        print(f"Comprehensive PLY file saved to: {output_file}")
        
        # Visualize the result with diffraction edges
        print("\nShowing result with diffraction edges. Close window to exit...")
        # Create line set for visualization
        line_points = []
        line_lines = []
        for (start_point, end_point), _ in diffraction_edges:
            line_points.extend([start_point, end_point])
            line_lines.append([len(line_points)-2, len(line_points)-1])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_lines)
        line_set.paint_uniform_color([1, 0, 0])  # Red color for edges
        
        # Show both point cloud and edges
        o3d.visualization.draw_geometries([pcd, line_set])
    else:
        print("\nNo diffraction edges found.")
        save_comprehensive_ply(all_points, all_colors, all_normals, 
                             all_labels, [], output_file)
        # Show original point cloud
        print("\nShowing point cloud. Close window to exit...")
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a segmented point cloud to find diffraction edges.")
    parser.add_argument("input_file", type=str, help="Path to the input segmented point cloud .ply file")
    parser.add_argument("output_file", type=str, help="Path to save the output .ply file with diffraction edges")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)