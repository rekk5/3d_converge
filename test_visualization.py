#!/usr/bin/env python3
"""
Test script for the 3D Pipeline Visualization Tool.

This script generates sample data and tests the visualization functionality.
"""

import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from visualization import PipelineVisualizer


def create_sample_point_cloud():
    """Create a sample point cloud for testing."""
    print("Creating sample point cloud...")
    
    # Create a simple cube point cloud
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    z = np.linspace(-1, 1, 20)
    
    # Create points on cube faces
    points = []
    colors = []
    
    # Front face (z = 1)
    for i in x:
        for j in y:
            points.append([i, j, 1])
            colors.append([1, 0, 0])  # Red
    
    # Back face (z = -1)
    for i in x:
        for j in y:
            points.append([i, j, -1])
            colors.append([0, 1, 0])  # Green
    
    # Left face (x = -1)
    for i in y:
        for j in z:
            points.append([-1, i, j])
            colors.append([0, 0, 1])  # Blue
    
    # Right face (x = 1)
    for i in y:
        for j in z:
            points.append([1, i, j])
            colors.append([1, 1, 0])  # Yellow
    
    # Top face (y = 1)
    for i in x:
        for j in z:
            points.append([i, 1, j])
            colors.append([1, 0, 1])  # Magenta
    
    # Bottom face (y = -1)
    for i in x:
        for j in z:
            points.append([i, -1, j])
            colors.append([0, 1, 1])  # Cyan
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    pcd.estimate_normals()
    
    return pcd


def create_sample_mesh():
    """Create a sample mesh for testing."""
    print("Creating sample mesh...")
    
    # Create a simple cube mesh
    vertices = np.array([
        # Front face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        # Back face
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]
    ])
    
    triangles = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [4, 6, 5], [4, 7, 6],
        # Left face
        [4, 5, 1], [4, 1, 0],
        # Right face
        [3, 2, 6], [3, 6, 7],
        # Top face
        [3, 7, 4], [3, 4, 0],
        # Bottom face
        [1, 5, 6], [1, 6, 2]
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Add colors
    colors = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],  # Red front
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]   # Green back
    ])
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    return mesh


def create_sample_mesh_with_edges():
    """Create a sample mesh with diffraction edges for testing."""
    print("Creating sample mesh with diffraction edges...")
    
    # Create two intersecting planes
    vertices = np.array([
        # Plane 1 (horizontal)
        [-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2],
        # Plane 2 (vertical)
        [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]
    ])
    
    triangles = np.array([
        # Plane 1
        [0, 1, 2], [0, 2, 3],
        # Plane 2
        [4, 6, 5], [4, 7, 6]
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Add colors
    colors = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],  # Red plane 1
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]   # Blue plane 2
    ])
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Create diffraction edges (intersection line)
    edges = [
        ([-1, 0, 0], [1, 0, 0])  # Edge along x-axis at intersection
    ]
    
    return mesh, edges


def save_ply_with_edges(mesh, edges, filename):
    """Save mesh with diffraction edges to PLY file."""
    print(f"Saving mesh with edges to {filename}...")
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertex_normals = np.asarray(mesh.vertex_normals)
    
    # Create vertex data
    vertex_data = []
    for v, c, n in zip(vertices, vertex_colors, vertex_normals):
        vertex_data.append((
            float(v[0]), float(v[1]), float(v[2]),
            int(c[0] * 255), int(c[1] * 255), int(c[2] * 255),
            float(n[0]), float(n[1]), float(n[2]),
            0, 1  # label, material
        ))
    
    # Create face data
    face_data = []
    for triangle in triangles:
        face_data.append(([int(triangle[0]), int(triangle[1]), int(triangle[2])],))
    
    # Create edge data
    edge_data = []
    for start_point, end_point in edges:
        edge_data.append((
            float(start_point[0]), float(start_point[1]), float(start_point[2]),
            float(end_point[0]), float(end_point[1]), float(end_point[2]),
            0.0, 1.0, 0.0,  # normal1
            1.0, 0.0, 0.0,  # normal2
            0, 1,  # plane1, plane2
            1, 2   # material1, material2
        ))
    
    # Define elements
    vertex = np.array(vertex_data,
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('label', 'u4'), ('material', 'u4')
        ])
    
    face = np.array(face_data,
        dtype=[('vertex_indices', 'i4', (3,))])
    
    edge = np.array(edge_data,
        dtype=[
            ('start_x', 'f4'), ('start_y', 'f4'), ('start_z', 'f4'),
            ('end_x', 'f4'), ('end_y', 'f4'), ('end_z', 'f4'),
            ('normal1_x', 'f4'), ('normal1_y', 'f4'), ('normal1_z', 'f4'),
            ('normal2_x', 'f4'), ('normal2_y', 'f4'), ('normal2_z', 'f4'),
            ('plane1', 'u4'), ('plane2', 'u4'),
            ('material1', 'u4'), ('material2', 'u4')
        ])
    
    # Save PLY file
    PlyData([
        PlyElement.describe(vertex, 'vertex'),
        PlyElement.describe(face, 'face'),
        PlyElement.describe(edge, 'edge')
    ], text=True).write(filename)


def test_visualization():
    """Test the visualization tool with sample data."""
    print("=== Testing 3D Pipeline Visualization Tool ===")
    
    # Create test directory
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate sample data
    point_cloud = create_sample_point_cloud()
    mesh = create_sample_mesh()
    mesh_with_edges, edges = create_sample_mesh_with_edges()
    
    # Save sample data
    point_cloud_path = os.path.join(test_dir, "sample_point_cloud.ply")
    mesh_path = os.path.join(test_dir, "sample_mesh.ply")
    mesh_with_edges_path = os.path.join(test_dir, "sample_mesh_with_edges.ply")
    
    o3d.io.write_point_cloud(point_cloud_path, point_cloud)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    save_ply_with_edges(mesh_with_edges, edges, mesh_with_edges_path)
    
    print(f"Sample data saved to {test_dir}/")
    
    # Test visualization
    print("\n=== Testing Visualization ===")
    
    # Create visualizer with corner view
    visualizer = PipelineVisualizer()
    visualizer.set_camera_params(
        position=[5, 5, 5],
        look_at=[0, 0, 0],
        up=[0, 1, 0],
        fov=75.0,
        zoom=0.8
    )
    
    # Test each visualization step
    print("\n1. Testing point cloud visualization...")
    try:
        visualizer.visualize_point_cloud(point_cloud_path, "Test Point Cloud")
        print("✓ Point cloud visualization successful")
    except Exception as e:
        print(f"✗ Point cloud visualization failed: {e}")
    
    print("\n2. Testing mesh visualization...")
    try:
        visualizer.visualize_mesh(mesh_path, "Test Mesh")
        print("✓ Mesh visualization successful")
    except Exception as e:
        print(f"✗ Mesh visualization failed: {e}")
    
    print("\n3. Testing mesh with edges visualization...")
    try:
        visualizer.visualize_mesh_with_edges(mesh_with_edges_path, "Test Mesh with Edges")
        print("✓ Mesh with edges visualization successful")
    except Exception as e:
        print(f"✗ Mesh with edges visualization failed: {e}")
    
    print("\n=== Test Complete ===")
    print(f"Sample data available in: {test_dir}/")
    print("You can now test the visualization tool with:")
    print(f"  python visualization.py --point_cloud {point_cloud_path}")
    print(f"  python visualization.py --mesh_with_edges {mesh_with_edges_path}")
    print(f"  python visualization.py --camera_position '8,8,8' --camera_fov 90")


if __name__ == "__main__":
    test_visualization() 