import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import argparse
from plyfile import PlyData, PlyElement


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
    
    def __init__(self, min_points=50, ransac_threshold=0.2, distance_threshold=0.1):
        """
        Initialize the detector with parameters.
        
        :param min_points: Minimum points required for line fitting
        :param ransac_threshold: Threshold for RANSAC plane fitting
        :param distance_threshold: Distance threshold for close points
        """
        self.min_points = min_points
        self.ransac_threshold = ransac_threshold
        self.distance_threshold = distance_threshold
        self.planes = []
        self.plane_normals = []
        self.diffraction_edges = []
        self.visualizer = None

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
        if angle < np.pi/15 or angle > 15*np.pi/16:
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
        """Fit a line to points using PCA."""
        if len(points) < self.min_points:
            return None, None
        
        pca = PCA(n_components=2)
        pca.fit(points)
        
        if pca.explained_variance_ratio_[0] < 0.9:
            return None, None
        
        if pca.explained_variance_ratio_[1] > 0.05:
            return None, None
        
        direction = pca.components_[0]
        centroid = np.mean(points, axis=0)
        
        projected_points = np.dot(points - centroid, direction[:, np.newaxis]) * direction + centroid
        distances = np.dot(projected_points - centroid, direction)
        
        min_idx, max_idx = np.argmin(distances), np.argmax(distances)
        edge_length = np.linalg.norm(projected_points[max_idx] - projected_points[min_idx])
        
        if edge_length < 0.5 or edge_length > 100.0:
            return None, None
        
        sorted_distances = np.sort(distances)
        gaps = sorted_distances[1:] - sorted_distances[:-1]
        if np.max(gaps) > edge_length * 0.6:
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
                                    if score > median_score * 0.5]

    def save_comprehensive_ply(self, output_file):
        """Save results to a comprehensive PLY file."""
        all_points = []
        all_colors = []
        all_normals = []
        all_labels = []
        
        for i, plane in enumerate(self.planes):
            points = np.asarray(plane.points)
            colors = np.asarray(plane.colors)
            normal, _ = self.fit_plane_ransac(points)
            if normal is None:
                normal = np.array([0, 0, 1])
            
            normals = np.tile(normal, (len(points), 1))
            labels = np.full(len(points), i)
            
            all_points.extend(points)
            all_colors.extend(colors)
            all_normals.extend(normals)
            all_labels.extend(labels)
        
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)
        all_normals = np.array(all_normals)
        all_labels = np.array(all_labels)
        
        # Create vertex data
        vertex_data = []
        for p, c, n, l in zip(all_points, all_colors, all_normals, all_labels):
            vertex_data.append((
                float(p[0]), float(p[1]), float(p[2]),
                int(c[0] * 255), int(c[1] * 255), int(c[2] * 255),
                float(n[0]), float(n[1]), float(n[2]),
                int(l), int(l)
            ))
        
        vertex = np.array(vertex_data,
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('label', 'u4'), ('material', 'u4')
            ])
        
        # Create edge data if we have edges
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
                PlyElement.describe(edge, 'edge')
            ], text=True).write(output_file)
        else:
            PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(output_file)

    def visualize_results(self):
        """Visualize the segmented planes and diffraction edges."""
        if not self.planes:
            print("No planes to visualize.")
            return

        self.visualizer = DiffractionEdgeVisualizer()

        # Add all planes to visualization
        combined_pcd = self.planes[0]
        for plane in self.planes[1:]:
            combined_pcd += plane
        self.visualizer.add_point_cloud(combined_pcd)

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

    def process(self, input_file, output_file, visualize=False):
        """
        Main processing pipeline.
        
        :param input_file: Path to input point cloud file
        :param output_file: Path to output file
        :param visualize: Whether to show visualization
        """
        print(f"\nProcessing {input_file}...")
        pcd = o3d.io.read_point_cloud(input_file)
        
        self.extract_planes_from_segmented_pcd(pcd)
        self.find_diffraction_edges()
        
        if self.diffraction_edges:
            print(f"\nFound {len(self.diffraction_edges)} diffraction edges")
        else:
            print("\nNo diffraction edges found.")
        
        self.save_comprehensive_ply(output_file)
        print(f"Results saved to: {output_file}")

        if visualize:
            self.visualize_results()


def main():
    parser = argparse.ArgumentParser(description="Process a segmented point cloud to find diffraction edges.")
    parser.add_argument("input_file", type=str, help="Path to the input segmented point cloud .ply file")
    parser.add_argument("output_file", type=str, help="Path to save the output .ply file with diffraction edges")
    parser.add_argument("--visualize", action="store_true", help="Show visualization of results")
    args = parser.parse_args()
    
    detector = DiffractionEdgeDetector()
    detector.process(args.input_file, args.output_file, args.visualize)


if __name__ == "__main__":
    main()