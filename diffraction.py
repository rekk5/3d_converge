import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import argparse
from plyfile import PlyData, PlyElement
import os
import sys


class DiffractionEdgeVisualizer:
    """Class for visualizing diffraction edges and point clouds."""
    
    def __init__(self, window_name="Diffraction Edge Visualization"):
        """
        Initialize the visualizer.
        
        :param window_name: Name of the visualization window
        """
        self.window_name = window_name
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        self.geometries = []

    def add_point_cloud(self, pcd, clear=False):
        """Add a point cloud to the visualization."""
        if clear:
            self.clear()
        self.geometries.append(pcd)
        self.vis.add_geometry(pcd)

    def add_line_set(self, points, lines, colors=None):
        """Add a line set to visualize edges."""
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        if colors is not None:
            line_set.colors = o3d.utility.Vector3dVector(colors)
        self.geometries.append(line_set)
        self.vis.add_geometry(line_set)

    def clear(self):
        """Clear all geometries from the visualization."""
        for geom in self.geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.geometries = []

    def update_view(self):
        """Update the visualization."""
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        """Run the visualizer."""
        self.vis.run()

    def destroy(self):
        """Destroy the visualization window."""
        self.vis.destroy_window()


class DiffractionEdgeDetector:
    """Class for detecting and visualizing diffraction edges in segmented point clouds."""
    
    def __init__(self, min_points=20, ransac_threshold=0.1, distance_threshold=0.15):
        """
        Initialize the detector with more lenient parameters.
        
        :param min_points: Reduced minimum points required for line fitting (was 50)
        :param ransac_threshold: Reduced threshold for RANSAC plane fitting (was 0.2)
        :param distance_threshold: Increased distance threshold for close points (was 0.1)
        """
        self.min_points = min_points
        self.ransac_threshold = ransac_threshold
        self.distance_threshold = distance_threshold
        self.planes = []
        self.plane_normals = []
        self.diffraction_edges = []
        self.visualizer = None
        self.meshes = []  # Add this to store generated meshes

    def fit_plane_ransac(self, points):
        """Fit a plane to points using RANSAC."""
        if len(points) < 3:
            return None, None
        
        try:
            ransac = RANSACRegressor(
                min_samples=3, 
                residual_threshold=self.ransac_threshold, 
                max_trials=20000
            )
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

    def find_close_points(self, points1, points2, normal1, normal2):
        """Find points close to the intersection of two planes."""
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
        if angle < np.pi/20 or angle > 19*np.pi/20:
            return np.array([])
        
        distances, indices = tree2.query(points1, k=1)
        mask = distances < self.distance_threshold
        
        close_points = []
        for point, dist, idx in zip(points1[mask], distances[mask], indices[mask]):
            dist1 = np.abs(np.dot(point - points1[0], normal1))
            dist2 = np.abs(np.dot(point - points2[0], normal2))
            
            if dist1 < self.distance_threshold and dist2 < self.distance_threshold:
                close_points.append(point)
        
        return np.array(close_points)

    def fit_line_to_points(self, points):
        """Fit a line to points using PCA with more lenient thresholds."""
        if len(points) < self.min_points:
            return None, None
        
        pca = PCA(n_components=2)
        pca.fit(points)
        
        if pca.explained_variance_ratio_[0] < 0.8:
            return None, None
        
        if pca.explained_variance_ratio_[1] > 0.1:
            return None, None
        
        direction = pca.components_[0]
        centroid = np.mean(points, axis=0)
        
        projected_points = np.dot(points - centroid, direction[:, np.newaxis]) * direction + centroid
        distances = np.dot(projected_points - centroid, direction)
        
        min_idx, max_idx = np.argmin(distances), np.argmax(distances)
        edge_length = np.linalg.norm(projected_points[max_idx] - projected_points[min_idx])
        
        if edge_length < 0.3 or edge_length > 500.0:
            return None, None
        
        sorted_distances = np.sort(distances)
        gaps = sorted_distances[1:] - sorted_distances[:-1]
        if np.max(gaps) > edge_length * 0.8:
            return None, None
        
        return projected_points[min_idx], projected_points[max_idx]

    def extract_planes_from_segmented_pcd(self, pcd):
        """Extract individual planes from segmented point cloud."""
        colors = np.asarray(pcd.colors)
        unique_colors = np.unique(colors, axis=0)
        
        for color in unique_colors:
            color_mask = np.all(colors == color, axis=1)
            indices = np.where(color_mask)[0]
            plane = pcd.select_by_index(indices)
            self.planes.append(plane)

    def create_mesh_from_plane(self, plane_points):
        """Create a mesh from plane points using Poisson surface reconstruction."""
        # Create a point cloud for the plane
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plane_points)
        
        # Estimate normals if they don't exist
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Create mesh using Ball Pivoting algorithm
        radii = [0.05, 0.1, 0.2, 0.4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        # Clean the mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh

    def find_diffraction_edges(self):
        """Find diffraction edges between planes."""
        edge_scores = []
        
        for i in range(len(self.planes)):
            for j in range(i + 1, len(self.planes)):
                points1 = np.asarray(self.planes[i].points)
                points2 = np.asarray(self.planes[j].points)
                
                if len(points1) < 3 or len(points2) < 3:
                    continue
                
                normal1, centroid1 = self.fit_plane_ransac(points1)
                normal2, centroid2 = self.fit_plane_ransac(points2)
                
                if normal1 is None or normal2 is None:
                    continue
                
                edge_direction = np.cross(normal1, normal2)
                if np.linalg.norm(edge_direction) < 1e-6:
                    continue
                
                edge_direction /= np.linalg.norm(edge_direction)
                line_centroid = (centroid1 + centroid2) / 2
                
                if np.dot(normal1, line_centroid - centroid1) < 0:
                    normal1 = -normal1
                if np.dot(normal2, line_centroid - centroid2) < 0:
                    normal2 = -normal2
                
                normal1 = -normal1
                normal2 = -normal2
                
                close_points = self.find_close_points(points1, points2, normal1, normal2)
                if len(close_points) > 1:
                    start_point, end_point = self.fit_line_to_points(close_points)
                    if start_point is not None and end_point is not None:
                        edge_length = np.linalg.norm(end_point - start_point)
                        point_density = len(close_points) / edge_length
                        score = point_density * edge_length
                        
                        self.diffraction_edges.append(((start_point, end_point), (i, j), normal1, normal2))
                        edge_scores.append(score)
        
        if edge_scores:
            median_score = np.median(edge_scores)
            self.diffraction_edges = [edge for edge, score in zip(self.diffraction_edges, edge_scores) 
                                    if score > median_score * 0.3]

    def process(self, input_file, output_file, visualize=False):
        """Main processing pipeline."""
        print(f"\nProcessing {input_file}...")
        pcd = o3d.io.read_point_cloud(input_file)
        
        self.extract_planes_from_segmented_pcd(pcd)
        
        # Create meshes from planes
        print("Generating meshes from planes...")
        for plane in self.planes:
            points = np.asarray(plane.points)
            mesh = self.create_mesh_from_plane(points)
            mesh.paint_uniform_color(np.asarray(plane.colors)[0])  # Use the plane's color
            self.meshes.append(mesh)
        
        self.find_diffraction_edges()
        
        if self.diffraction_edges:
            print(f"\nFound {len(self.diffraction_edges)} diffraction edges")
        else:
            print("\nNo diffraction edges found.")
        
        self.save_comprehensive_ply(output_file)
        print(f"Results saved to: {output_file}")

        if visualize:
            self.visualize_results()

    def save_comprehensive_ply(self, output_file):
        """Save results to a comprehensive PLY file with meshes."""
        all_vertices = []
        all_triangles = []
        all_vertex_colors = []
        all_vertex_normals = []
        all_labels = []
        vertex_offset = 0
        
        # Process each mesh
        for i, mesh in enumerate(self.meshes):
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            vertex_colors = np.asarray(mesh.vertex_colors)
            vertex_normals = np.asarray(mesh.vertex_normals)
            
            # Add vertices with offset
            all_vertices.extend(vertices)
            all_triangles.extend(triangles + vertex_offset)
            all_vertex_colors.extend(vertex_colors)
            all_vertex_normals.extend(vertex_normals)
            all_labels.extend([i] * len(vertices))
            
            vertex_offset += len(vertices)
        
        # Create vertex data
        vertex_data = []
        for v, c, n, l in zip(all_vertices, all_vertex_colors, all_vertex_normals, all_labels):
            vertex_data.append((
                float(v[0]), float(v[1]), float(v[2]),
                int(c[0] * 255), int(c[1] * 255), int(c[2] * 255),
                float(n[0]), float(n[1]), float(n[2]),
                int(l), int(l)
            ))
        
        # Create face data
        face_data = []
        for triangle in all_triangles:
            face_data.append(([int(triangle[0]), int(triangle[1]), int(triangle[2])],))
        
        # Define vertex and face elements
        vertex = np.array(vertex_data,
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('label', 'u4'), ('material', 'u4')
            ])
        
        face = np.array(face_data,
            dtype=[('vertex_indices', 'i4', (3,))])
        
        # Add edge data if we have edges
        if self.diffraction_edges:
            edge_data = []
            for (start_point, end_point), (plane1_idx, plane2_idx), normal1, normal2 in self.diffraction_edges:
                edge_data.append((
                    float(start_point[0]), float(start_point[1]), float(start_point[2]),
                    float(end_point[0]), float(end_point[1]), float(end_point[2]),
                    float(normal1[0]), float(normal1[1]), float(normal1[2]),
                    float(normal2[0]), float(normal2[1]), float(normal2[2]),
                    int(plane1_idx), int(plane2_idx),
                    int(plane1_idx), int(plane2_idx)
                ))
            
            edge = np.array(edge_data,
                dtype=[
                    ('start_x', 'f4'), ('start_y', 'f4'), ('start_z', 'f4'),
                    ('end_x', 'f4'), ('end_y', 'f4'), ('end_z', 'f4'),
                    ('normal1_x', 'f4'), ('normal1_y', 'f4'), ('normal1_z', 'f4'),
                    ('normal2_x', 'f4'), ('normal2_y', 'f4'), ('normal2_z', 'f4'),
                    ('plane1', 'u4'), ('plane2', 'u4'),
                    ('material1', 'u4'), ('material2', 'u4')
                ])
            
            PlyData([
                PlyElement.describe(vertex, 'vertex'),
                PlyElement.describe(face, 'face'),
                PlyElement.describe(edge, 'edge')
            ], text=True).write(output_file)
        else:
            PlyData([
                PlyElement.describe(vertex, 'vertex'),
                PlyElement.describe(face, 'face')
            ], text=True).write(output_file)

    def visualize_results(self):
        """Visualize the segmented meshes and diffraction edges."""
        if not self.meshes:
            print("No meshes to visualize.")
            return

        self.visualizer = DiffractionEdgeVisualizer()

        # Add all meshes to visualization
        for mesh in self.meshes:
            self.visualizer.vis.add_geometry(mesh)

        # Visualize diffraction edges
        if self.diffraction_edges:
            edge_points = []
            edge_lines = []
            edge_colors = []
            
            for idx, ((start_point, end_point), _, _, _) in enumerate(self.diffraction_edges):
                edge_points.extend([start_point, end_point])
                edge_lines.append([2*idx, 2*idx+1])
                edge_colors.append([1, 0, 0])  # Red color for edges
            
            if edge_points:
                self.visualizer.add_line_set(
                    np.array(edge_points),
                    np.array(edge_lines),
                    np.array(edge_colors)
                )

        # Set default camera view
        self.visualizer.vis.get_view_control().set_zoom(0.8)
        self.visualizer.run()
        self.visualizer.destroy()


def validate_segmented_point_cloud(file_path):
    """
    Validates that the provided file exists and is properly segmented for diffraction edge detection.
    
    Args:
        file_path: Path to the segmented point cloud file
        
    Returns:
        bool: True if the file is valid, False otherwise
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.ply']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        print(f"Error: Only PLY files are supported for diffraction edge detection.")
        return False
    
    # Try to open the file with Open3D
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Check if there are enough points
        if len(np.asarray(pcd.points)) < 100:
            print(f"Error: Point cloud has too few points ({len(np.asarray(pcd.points))}).")
            return False
        
        # Check if the point cloud has color information (needed for segmentation)
        if not pcd.has_colors():
            print("Error: Point cloud does not have color information. Input must be a segmented point cloud.")
            return False
        
        # Check if there are multiple segments (different colors)
        colors = np.asarray(pcd.colors)
        unique_colors = np.unique(colors, axis=0)
        if len(unique_colors) < 2:
            print(f"Error: Point cloud has only {len(unique_colors)} segment(s). At least 2 segments are needed for edge detection.")
            return False
            
        print(f"Point cloud validated: {len(np.asarray(pcd.points))} points, {len(unique_colors)} segments")
        return True
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process a segmented point cloud to find diffraction edges.")
    parser.add_argument("input_file", type=str, help="Path to the input segmented point cloud .ply file")
    parser.add_argument("output_file", type=str, help="Path to save the output .ply file with diffraction edges")
    parser.add_argument("--visualize", action="store_true", help="Show visualization of results")
    args = parser.parse_args()
    
    # Validate input file
    if not validate_segmented_point_cloud(args.input_file):
        sys.exit(1)
    
    detector = DiffractionEdgeDetector()
    detector.process(args.input_file, args.output_file, args.visualize)


if __name__ == "__main__":
    main()