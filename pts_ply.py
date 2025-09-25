import open3d as o3d
import numpy as np
import sys

def process_pts_file(input_path, output_path, batch_size=100000):
    """Process a large .pts file in batches and write to a .ply file."""
    points = []
    colors = []
    try:
        with open(input_path, 'r') as file:
            # Skip the first 8 lines of metadata
            for _ in range(8):
                next(file)
            
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[:3])
                    r, g, b = map(int, parts[4:7])
                    points.append([x, y, z])
                    colors.append([r, g, b])

                # Process in batches to manage memory
                if len(points) >= batch_size:
                    write_batch_to_ply(points, colors, output_path, append=True)
                    points.clear()
                    colors.clear()

            # Write any remaining points
            if points:
                write_batch_to_ply(points, colors, output_path, append=True)

        print(f"Successfully wrote to {output_path}")
    except Exception as e:
        print(f"Error processing .pts file: {e}")
        sys.exit(1)

def write_batch_to_ply(points, colors, output_path, append=False):
    """Write a batch of points and colors to a .ply file."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)

    # Write the point cloud to the output file
    if append:
        # Append mode: read existing points and combine
        existing_pcd = o3d.io.read_point_cloud(output_path) if o3d.io.read_point_cloud(output_path) else o3d.geometry.PointCloud()
        combined_pcd = existing_pcd + point_cloud
        o3d.io.write_point_cloud(output_path, combined_pcd, write_ascii=True, print_progress=True)
    else:
        o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=True, print_progress=True)

def main(input_path, output_path):
    process_pts_file(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pts_ply_conversion.py <input.pts> <output.ply>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)