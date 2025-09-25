import open3d as o3d
import numpy as np
import argparse
import os
import sys
from plyfile import PlyData
import matplotlib.pyplot as plt
from comparison import load_differential_ply


class PipelineVisualizer:
    """Comprehensive visualizer for the 3D reconstruction pipeline steps."""
    
    def __init__(self, window_name="3D Pipeline Visualization"):
        """
        Initialize the visualizer.
        
        :param window_name: Name of the visualization window
        """
        self.window_name = window_name
        self.vis = None
        self.geometries = []
        self.camera_params = {
            'position': [0, 0, 5],  # Default camera position
            'look_at': [0, 0, 0],   # Default look at point
            'up': [0, 1, 0],        # Default up vector
            'fov': 60.0,            # Default field of view in degrees
            'zoom': 0.8             # Default zoom level
        }
        
    def set_camera_params(self, position=None, look_at=None, up=None, fov=None, zoom=None):
        """
        Set camera parameters for visualization.
        
        :param position: Camera position [x, y, z]
        :param look_at: Point to look at [x, y, z]
        :param up: Up vector [x, y, z]
        :param fov: Field of view in degrees (note: Open3D doesn't support FOV control directly)
        :param zoom: Zoom level (used to approximate FOV control)
        """
        if position is not None:
            self.camera_params['position'] = position
        if look_at is not None:
            self.camera_params['look_at'] = look_at
        if up is not None:
            self.camera_params['up'] = up
        if fov is not None:
            self.camera_params['fov'] = fov
            # Adjust zoom based on FOV to approximate the desired field of view
            if zoom is None:
                # Convert FOV to zoom approximation
                # Wider FOV = smaller zoom, narrower FOV = larger zoom
                fov_rad = np.radians(fov)
                self.camera_params['zoom'] = 1.0 / np.tan(fov_rad / 2) * 0.5
        if zoom is not None:
            self.camera_params['zoom'] = zoom
    
    def create_visualizer(self):
        """Create and configure the Open3D visualizer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=1200, height=800)
        
        # Set render options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
        opt.point_size = 2.0
        opt.line_width = 3.0
        opt.show_coordinate_frame = True
        
        # Set camera parameters
        ctr = self.vis.get_view_control()
        
        # Calculate front vector from position and look_at
        position = np.array(self.camera_params['position'])
        look_at = np.array(self.camera_params['look_at'])
        front = look_at - position
        front = front / np.linalg.norm(front)
        
        ctr.set_front(front)
        ctr.set_lookat(look_at)
        ctr.set_up(self.camera_params['up'])
        ctr.set_zoom(self.camera_params['zoom'])
        
        # Note: Open3D doesn't have set_field_of_view method
        # FOV is typically controlled by the projection matrix
        # We'll use zoom to approximate FOV control
    
    def add_geometry(self, geometry, clear=False):
        """Add geometry to the visualization."""
        if clear:
            self.clear()
        self.geometries.append(geometry)
        self.vis.add_geometry(geometry)
    
    def clear(self):
        """Clear all geometries from the visualization."""
        if self.vis is not None:
            for geom in self.geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.geometries = []
    
    def update_view(self):
        """Update the visualization."""
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def run(self):
        """Run the visualizer."""
        if self.vis is not None:
            self.vis.run()
    
    def destroy(self):
        """Destroy the visualization window."""
        if self.vis is not None:
            self.vis.destroy_window()
    
    def load_point_cloud(self, file_path):
        """
        Load a point cloud from file.
        
        :param file_path: Path to the point cloud file
        :return: Loaded point cloud or None if failed
        """
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(np.asarray(pcd.points)) == 0:
                print(f"Error: Point cloud file is empty: {file_path}")
                return None
            print(f"Loaded point cloud: {len(np.asarray(pcd.points))} points")
            return pcd
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return None
    
    def load_mesh_with_edges(self, file_path):
        """
        Load a mesh with diffraction edges from PLY file.
        
        :param file_path: Path to the PLY file
        :return: Tuple of (meshes, edges) or (None, None) if failed
        """
        try:
            ply_data = PlyData.read(file_path)
            
            meshes = []
            edges = []
            
            # Check if we have vertex and face data (mesh)
            if 'vertex' in ply_data and 'face' in ply_data:
                vertex_data = ply_data['vertex']
                face_data = ply_data['face']
                
                # Extract vertex information
                vertices = np.column_stack([
                    vertex_data['x'],
                    vertex_data['y'],
                    vertex_data['z']
                ])
                
                colors = np.column_stack([
                    vertex_data['red'],
                    vertex_data['green'],
                    vertex_data['blue']
                ]) / 255.0
                
                normals = np.column_stack([
                    vertex_data['nx'],
                    vertex_data['ny'],
                    vertex_data['nz']
                ])
                
                # Extract face information
                faces = np.asarray(face_data['vertex_indices'])
                
                # Create mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
                
                # Compute normals if not present
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                
                meshes.append(mesh)
                print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            
            # Check if we have edge data (diffraction edges)
            if 'edge' in ply_data:
                edge_data = ply_data['edge']
                
                for edge in edge_data:
                    start_point = np.array([edge['start_x'], edge['start_y'], edge['start_z']])
                    end_point = np.array([edge['end_x'], edge['end_y'], edge['end_z']])
                    edges.append((start_point, end_point))
                
                print(f"Loaded {len(edges)} diffraction edges")
            
            return meshes, edges
            
        except Exception as e:
            print(f"Error loading mesh with edges: {e}")
            return None, None
    
    def visualize_point_cloud(self, file_path, title="Point Cloud Visualization"):
        """
        Visualize a point cloud.
        
        :param file_path: Path to the point cloud file
        :param title: Title for the visualization window
        """
        print(f"\n=== {title} ===")
        
        pcd = self.load_point_cloud(file_path)
        if pcd is None:
            return
        
        self.create_visualizer()
        self.add_geometry(pcd, clear=True)
        
        print("Point cloud visualization ready. Press 'Q' to exit.")
        self.run()
        self.destroy()
    
    def visualize_mesh(self, file_path, title="Mesh Visualization"):
        """
        Visualize a mesh.
        
        :param file_path: Path to the mesh file
        :param title: Title for the visualization window
        """
        print(f"\n=== {title} ===")
        
        meshes, edges = self.load_mesh_with_edges(file_path)
        if meshes is None:
            return
        
        self.create_visualizer()
        
        # Add all meshes
        for i, mesh in enumerate(meshes):
            self.add_geometry(mesh, clear=(i == 0))
        
        print("Mesh visualization ready. Press 'Q' to exit.")
        self.run()
        self.destroy()
    
    def visualize_mesh_with_edges(self, file_path, title="Mesh with Diffraction Edges"):
        """
        Visualize a mesh with diffraction edges.
        
        :param file_path: Path to the mesh file with edges
        :param title: Title for the visualization window
        """
        print(f"\n=== {title} ===")
        
        meshes, edges = self.load_mesh_with_edges(file_path)
        if meshes is None:
            return
        
        self.create_visualizer()
        
        # Add all meshes
        for i, mesh in enumerate(meshes):
            self.add_geometry(mesh, clear=(i == 0))
        
        # Add diffraction edges as line sets
        if edges:
            edge_points = []
            edge_lines = []
            
            for idx, (start_point, end_point) in enumerate(edges):
                edge_points.extend([start_point, end_point])
                edge_lines.append([2*idx, 2*idx+1])
            
            if edge_points:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.array(edge_points))
                line_set.lines = o3d.utility.Vector2iVector(np.array(edge_lines))
                line_set.paint_uniform_color([1, 0, 0])  # Red color for edges
                self.add_geometry(line_set)
        
        print("Mesh with diffraction edges visualization ready. Press 'Q' to exit.")
        self.run()
        self.destroy()
    
    def visualize_pipeline_steps(self, point_cloud_path=None, mesh_path=None, 
                                mesh_with_edges_path=None, auto_advance=False):
        """
        Visualize all pipeline steps in sequence.
        
        :param point_cloud_path: Path to point cloud file
        :param mesh_path: Path to mesh file
        :param mesh_with_edges_path: Path to mesh with edges file
        :param auto_advance: Whether to automatically advance between steps
        """
        print("\n=== 3D Pipeline Visualization ===")
        print("Camera parameters:")
        print(f"  Position: {self.camera_params['position']}")
        print(f"  Look at: {self.camera_params['look_at']}")
        print(f"  Up vector: {self.camera_params['up']}")
        print(f"  FOV: {self.camera_params['fov']}°")
        print(f"  Zoom: {self.camera_params['zoom']}")
        
        steps = []
        
        if point_cloud_path and os.path.exists(point_cloud_path):
            steps.append(('Point Cloud', point_cloud_path, self.visualize_point_cloud))
        
        if mesh_path and os.path.exists(mesh_path):
            steps.append(('Mesh', mesh_path, self.visualize_mesh))
        
        if mesh_with_edges_path and os.path.exists(mesh_with_edges_path):
            steps.append(('Mesh with Diffraction Edges', mesh_with_edges_path, self.visualize_mesh_with_edges))
        
        if not steps:
            print("No valid files found for visualization.")
            return
        
        print(f"\nFound {len(steps)} steps to visualize:")
        for i, (name, path, _) in enumerate(steps):
            print(f"  {i+1}. {name}: {path}")
        
        if auto_advance:
            print("\nAuto-advancing through steps...")
            for name, path, viz_func in steps:
                print(f"\nVisualizing: {name}")
                viz_func(path, f"Step: {name}")
                if auto_advance and name != steps[-1][0]:
                    input("Press Enter to continue to next step...")
        else:
            print("\nChoose visualization step:")
            for i, (name, _, _) in enumerate(steps):
                print(f"  {i+1}. {name}")
            print("  0. Exit")
            
            while True:
                try:
                    choice = input("\nEnter choice (0-{}): ".format(len(steps)))
                    choice = int(choice)
                    
                    if choice == 0:
                        break
                    elif 1 <= choice <= len(steps):
                        name, path, viz_func = steps[choice - 1]
                        viz_func(path, f"Step: {name}")
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break


def parse_camera_params(args):
    """
    Parse camera parameters from command line arguments.
    
    :param args: Parsed command line arguments
    :return: Dictionary of camera parameters
    """
    camera_params = {}
    
    if args.camera_position:
        camera_params['position'] = [float(x) for x in args.camera_position.split(',')]
    
    if args.camera_look_at:
        camera_params['look_at'] = [float(x) for x in args.camera_look_at.split(',')]
    
    if args.camera_up:
        camera_params['up'] = [float(x) for x in args.camera_up.split(',')]
    
    if args.camera_fov:
        camera_params['fov'] = float(args.camera_fov)
    
    if args.camera_zoom:
        camera_params['zoom'] = float(args.camera_zoom)
    
    return camera_params


def main():
    parser = argparse.ArgumentParser(description='3D Pipeline Visualization Tool')
    
    # Input files
    parser.add_argument('--point_cloud', type=str, help='Path to point cloud file (.ply)')
    parser.add_argument('--mesh', type=str, help='Path to mesh file (.ply)')
    parser.add_argument('--mesh_with_edges', type=str, help='Path to mesh with diffraction edges (.ply)')
    
    # Camera parameters
    parser.add_argument('--camera_position', type=str, default='5,5,5',
                        help='Camera position as "x,y,z" (default: "5,5,5")')
    parser.add_argument('--camera_look_at', type=str, default='0,0,0',
                        help='Camera look at point as "x,y,z" (default: "0,0,0")')
    parser.add_argument('--camera_up', type=str, default='0,1,0',
                        help='Camera up vector as "x,y,z" (default: "0,0,1")')
    parser.add_argument('--camera_fov', type=float, default=60.0,
                        help='Camera field of view in degrees (default: 60.0)')
    parser.add_argument('--camera_zoom', type=float, default=0.8,
                        help='Camera zoom level (default: 0.8)')
    
    # Visualization options
    parser.add_argument('--auto_advance', action='store_true',
                        help='Automatically advance through visualization steps')
    parser.add_argument('--step', type=str, choices=['point_cloud', 'mesh', 'mesh_with_edges'],
                        help='Visualize specific step only')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PipelineVisualizer()
    
    # Set camera parameters
    camera_params = parse_camera_params(args)
    visualizer.set_camera_params(**camera_params)
    
    # Determine which files to use
    point_cloud_path = args.point_cloud
    mesh_path = args.mesh
    mesh_with_edges_path = args.mesh_with_edges
    
    # If no specific files provided, try to find them in current directory
    if not any([point_cloud_path, mesh_path, mesh_with_edges_path]):
        print("No files specified. Looking for common pipeline output files...")
        
        # Look for common output files
        common_files = {
            'point_cloud': ['point_cloud.ply', 'recon.ply', 'dense.ply'],
            'mesh': ['mesh.ply', 'saalas_mesh.ply'],
            'mesh_with_edges': ['saalas_mesh.ply', 'mesh_with_edges.ply']
        }
        
        for file_type, candidates in common_files.items():
            for candidate in candidates:
                if os.path.exists(candidate):
                    if file_type == 'point_cloud' and not point_cloud_path:
                        point_cloud_path = candidate
                        print(f"Found point cloud: {candidate}")
                    elif file_type == 'mesh' and not mesh_path:
                        mesh_path = candidate
                        print(f"Found mesh: {candidate}")
                    elif file_type == 'mesh_with_edges' and not mesh_with_edges_path:
                        mesh_with_edges_path = candidate
                        print(f"Found mesh with edges: {candidate}")
                    break
    
    # Visualize specific step if requested
    if args.step:
        if args.step == 'point_cloud' and point_cloud_path:
            visualizer.visualize_point_cloud(point_cloud_path)
        elif args.step == 'mesh' and mesh_path:
            visualizer.visualize_mesh(mesh_path)
        elif args.step == 'mesh_with_edges' and mesh_with_edges_path:
            visualizer.visualize_mesh_with_edges(mesh_with_edges_path)
        else:
            print(f"Error: {args.step} file not found or not specified.")
            sys.exit(1)
    else:
        # Visualize all available steps
        visualizer.visualize_pipeline_steps(
            point_cloud_path=point_cloud_path,
            mesh_path=mesh_path,
            mesh_with_edges_path=mesh_with_edges_path,
            auto_advance=args.auto_advance
        )


if __name__ == "__main__":
    main() 