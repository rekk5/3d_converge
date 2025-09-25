import numpy as np
from plyfile import PlyData
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_diffraction_edge_normals(edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for e in edges:
        # Start and end points
        start = np.array([e['start_x'], e['start_y'], e['start_z']])
        end = np.array([e['end_x'], e['end_y'], e['end_z']])
        
        # Normals
        normal1 = np.array([e['normal1_x'], e['normal1_y'], e['normal1_z']])
        normal2 = np.array([e['normal2_x'], e['normal2_y'], e['normal2_z']])
        
        # Debugging: Print the data being plotted
        print(f"Edge from {start} to {end}")
        print(f"Normal 1: {normal1}, Normal 2: {normal2}")
        
        # Plot the edge
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-', label='Edge')
        
        # Plot the normals
        ax.quiver(start[0], start[1], start[2], normal1[0], normal1[1], normal1[2], color='r', length=0.1, normalize=True, label='Normal 1')
        ax.quiver(start[0], start[1], start[2], normal2[0], normal2[1], normal2[2], color='b', length=0.1, normalize=True, label='Normal 2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Diffraction Edge Normals')
    plt.show()

def check_ply_contents(ply_file):
    try:
        print(f"\nAnalyzing PLY file: {ply_file}")
        print("-" * 50)
        
        # Load PLY file
        ply_data = PlyData.read(ply_file)
        
        # Define required properties
        required_vertex_properties = {
            'x': 'f4', 'y': 'f4', 'z': 'f4',           # position
            'red': 'u1', 'green': 'u1', 'blue': 'u1',   # color
            'nx': 'f4', 'ny': 'f4', 'nz': 'f4',         # normal
            'label': 'u4', 'material': 'u4'             # metadata
        }
        
        required_edge_properties = {
            'start_x': 'f4', 'start_y': 'f4', 'start_z': 'f4',  # start point
            'end_x': 'f4', 'end_y': 'f4', 'end_z': 'f4',       # end point
            'normal1_x': 'f4', 'normal1_y': 'f4', 'normal1_z': 'f4',  # normal1
            'normal2_x': 'f4', 'normal2_y': 'f4', 'normal2_z': 'f4',  # normal2
            'plane1': 'u4', 'plane2': 'u4',                      # plane indices
            'material1': 'u4', 'material2': 'u4'                 # material indices
        }
        
        # Check vertex data
        if 'vertex' in ply_data:
            vertex = ply_data['vertex']
            print("\nVertex Data:")
            print(f"Number of vertices: {len(vertex)}")
            
            # Check for required vertex properties
            print("\nChecking required vertex properties:")
            vertex_properties = {prop.name: prop.val_dtype for prop in vertex.properties}
            for prop_name, prop_type in required_vertex_properties.items():
                if prop_name in vertex_properties:
                    status = "✓" if vertex_properties[prop_name] == prop_type else "⚠ (wrong type)"
                else:
                    status = "✗"
                print(f"- {prop_name}: {status}")
            
            print("\nAvailable properties:")
            for prop in vertex.properties:
                print(f"- {prop.name}: {prop.val_dtype}")
            
            # Print sample of vertex data
            print("\nSample vertex data (first 3 points):")
            for i in range(min(3, len(vertex))):
                print(f"\nPoint {i+1}:")
                for prop in vertex.properties:
                    value = vertex[prop.name][i]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    print(f"  {prop.name}: {value}")
        
        # Check edge data
        if 'edge' in ply_data:
            edge = ply_data['edge']
            print("\nEdge Data:")
            print(f"Number of edges: {len(edge)}")
            
            # Verify edge properties
            for prop, dtype in required_edge_properties.items():
                if prop in edge.data.dtype.names:
                    print(f"Property '{prop}' found with type '{edge.data.dtype[prop]}'")
                else:
                    print(f"Property '{prop}' is missing!")
            
            # Print sample of edge data
            print("\nSample edge data (first 3 edges):")
            for i in range(min(3, len(edge))):
                print(f"\nEdge {i+1}:")
                for prop in edge.properties:
                    value = edge[prop.name][i]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    print(f"  {prop.name}: {value}")
        
        # Print unique labels if they exist
        if 'vertex' in ply_data and 'label' in vertex.properties:
            unique_labels = np.unique(vertex['label'])
            print(f"\nUnique plane labels found: {unique_labels}")
            print(f"Total number of planes: {len(unique_labels)}")
            
            # Count points per label
            print("\nPoints per plane:")
            for label in unique_labels:
                count = np.sum(vertex['label'] == label)
                print(f"Plane {label}: {count} points")
        
        # Check for diffraction edges
        if 'edge' in ply_data:
            edge = ply_data['edge']
            if 'plane1' in edge.properties and 'plane2' in edge.properties:
                print("\nDiffraction edge information:")
                unique_plane_pairs = set()
                for e in edge:
                    unique_plane_pairs.add((e['plane1'], e['plane2']))
                print(f"Number of unique plane pairs forming edges: {len(unique_plane_pairs)}")
                print("\nPlane pairs and their properties:")
                for e in edge[:3]:  # Show first 3 edges as examples
                    print(f"\nEdge between planes {e['plane1']} and {e['plane2']}:")
                    print(f"  Start point: ({e['start_x']:.3f}, {e['start_y']:.3f}, {e['start_z']:.3f})")
                    print(f"  End point: ({e['end_x']:.3f}, {e['end_y']:.3f}, {e['end_z']:.3f})")
                    print(f"  Normal 1: ({e['normal1_x']:.3f}, {e['normal1_y']:.3f}, {e['normal1_z']:.3f})")
                    print(f"  Normal 2: ({e['normal2_x']:.3f}, {e['normal2_y']:.3f}, {e['normal2_z']:.3f})")
                    print(f"  Materials: {e['material1']} and {e['material2']}")
                
                # Visualize diffraction edge normals
                plot_diffraction_edge_normals(edge)
        
    except Exception as e:
        print(f"\nError reading PLY file: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description="Check contents of a PLY file with segmentation and diffraction edge data.")
    parser.add_argument("ply_file", help="Path to the PLY file to analyze")
    args = parser.parse_args()
    
    check_ply_contents(args.ply_file)

if __name__ == "__main__":
    main()