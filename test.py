import numpy as np
from plyfile import PlyData
import argparse

def check_ply_contents(ply_file):
    try:
        print(f"\nAnalyzing PLY file: {ply_file}")
        print("-" * 50)
        
        # Load PLY file
        ply_data = PlyData.read(ply_file)
        
        # Check vertex data
        if 'vertex' in ply_data:
            vertex = ply_data['vertex']
            print("\nVertex Data:")
            print(f"Number of vertices: {len(vertex)}")
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
            print("\nAvailable properties:")
            for prop in edge.properties:
                print(f"- {prop.name}: {prop.val_dtype}")
            
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
            if 'plane1' in edge.properties and 'plane2' in edge.properties:
                print("\nDiffraction edge information:")
                unique_plane_pairs = set()
                for e in edge:
                    unique_plane_pairs.add((e['plane1'], e['plane2']))
                print(f"Number of unique plane pairs forming edges: {len(unique_plane_pairs)}")
                print("Plane pairs:")
                for pair in unique_plane_pairs:
                    print(f"Planes {pair[0]} and {pair[1]}")
        
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