import numpy as np
import open3d as o3d

class PointCloudConnector:
    def __init__(self, pcd):
        """Initialize with a pre-segmented point cloud"""
        self.original_pcd = pcd
        self.segments = {}
        self.connections = {}
        self.combined_group = []  # Track segments that are connected
        self._separate_segments()
        
    def _separate_segments(self):
        """Separate the point cloud into segments based on colors"""
        points = np.asarray(self.original_pcd.points)
        colors = np.asarray(self.original_pcd.colors)
        normals = np.asarray(self.original_pcd.normals)
        
        # Find unique colors
        unique_colors = np.unique(colors, axis=0)
        
        print("\nAvailable segments:")
        for i, color in enumerate(unique_colors):
            # Find points with this color
            color_mask = np.all(colors == color, axis=1)
            segment_points = points[color_mask]
            segment_normals = normals[color_mask]
            
            # Calculate bounds
            min_bound = np.min(segment_points, axis=0)
            max_bound = np.max(segment_points, axis=0)
            
            # Create new point cloud for this segment
            pcd_segment = o3d.geometry.PointCloud()
            pcd_segment.points = o3d.utility.Vector3dVector(segment_points)
            pcd_segment.normals = o3d.utility.Vector3dVector(segment_normals)
            pcd_segment.colors = o3d.utility.Vector3dVector(np.tile(color, (len(segment_points), 1)))
            
            segment_id = f"segment_{i}"
            self.segments[segment_id] = pcd_segment
            self.connections[segment_id] = set()
            
            # Print segment info with bounds
            print(f"{segment_id}:")
            print(f"  Points: {len(segment_points)}")
            print(f"  Color: {color}")
            print(f"  Bounds:")
            print(f"    Min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
            print(f"    Max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
    
    def connect_segments(self, segment_id1, segment_id2):
        """Connect two segments"""
        if segment_id1 in self.segments and segment_id2 in self.segments:
            self.connections[segment_id1].add(segment_id2)
            self.connections[segment_id2].add(segment_id1)
            
            # Update combined group
            if not self.combined_group:
                self.combined_group = [segment_id1, segment_id2]
            elif segment_id1 in self.combined_group and segment_id2 not in self.combined_group:
                self.combined_group.append(segment_id2)
            elif segment_id2 in self.combined_group and segment_id1 not in self.combined_group:
                self.combined_group.append(segment_id1)
            return True
        return False
    
    def visualize_segment(self, segment_id):
        """Visualize a single segment"""
        if segment_id in self.segments:
            o3d.visualization.draw_geometries([self.segments[segment_id]])
        else:
            print(f"Segment {segment_id} not found")
    
    def visualize_all(self):
        """Visualize all segments"""
        o3d.visualization.draw_geometries(list(self.segments.values()))
    
    def connect_and_visualize(self, segment_id1, segment_id2):
        """Connect two segments and visualize the result"""
        if self.connect_segments(segment_id1, segment_id2):
            print(f"Connected {segment_id1} to {segment_id2}")
            print(f"Current combined group: {self.combined_group}")
            
            # Visualize the combined group
            combined_segments = [self.segments[seg_id] for seg_id in self.combined_group]
            o3d.visualization.draw_geometries(combined_segments)
        else:
            print("Connection failed. Check segment IDs.")

    def get_combined_group(self):
        """Return the current combined group"""
        return self.combined_group
    
    def translate_segment(self, segment_id, translation):
        """Translate a segment by a given vector [x, y, z]"""
        if segment_id in self.segments:
            self.segments[segment_id].translate(translation)
            print(f"Translated {segment_id} by {translation}")
            # Visualize the result
            self.visualize_current_state()
            return True
        print(f"Segment {segment_id} not found")
        return False

    def rotate_segment(self, segment_id, angle_deg, axis='z'):
        """Rotate a segment by angle (degrees) around specified axis"""
        if segment_id not in self.segments:
            print(f"Segment {segment_id} not found")
            return False

        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)
        
        # Create rotation matrix
        if axis.lower() == 'x':
            R = self.segments[segment_id].get_rotation_matrix_from_xyz((angle_rad, 0, 0))
        elif axis.lower() == 'y':
            R = self.segments[segment_id].get_rotation_matrix_from_xyz((0, angle_rad, 0))
        else:  # z-axis
            R = self.segments[segment_id].get_rotation_matrix_from_xyz((0, 0, angle_rad))
        
        # Apply rotation
        center = self.segments[segment_id].get_center()
        self.segments[segment_id].translate(-center)
        self.segments[segment_id].rotate(R, center=(0, 0, 0))
        self.segments[segment_id].translate(center)
        
        print(f"Rotated {segment_id} by {angle_deg} degrees around {axis}-axis")
        self.visualize_current_state()
        return True

    def remove_segment(self, segment_id):
        """Remove a segment from the point cloud"""
        if segment_id in self.segments:
            # Remove from segments
            del self.segments[segment_id]
            # Remove from connections
            del self.connections[segment_id]
            for seg in self.connections:
                self.connections[seg].discard(segment_id)
            # Remove from combined group
            if segment_id in self.combined_group:
                self.combined_group.remove(segment_id)
            
            print(f"Removed {segment_id}")
            self.visualize_current_state()
            return True
        print(f"Segment {segment_id} not found")
        return False

    def crop_segment(self, segment_id, min_bound, max_bound):
        """Crop a segment to keep only points within the specified bounds"""
        if segment_id not in self.segments:
            print(f"Segment {segment_id} not found")
            return False

        points = np.asarray(self.segments[segment_id].points)
        colors = np.asarray(self.segments[segment_id].colors)
        normals = np.asarray(self.segments[segment_id].normals)

        # Create mask for points within bounds
        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        
        # Create new point cloud with cropped points
        pcd_cropped = o3d.geometry.PointCloud()
        pcd_cropped.points = o3d.utility.Vector3dVector(points[mask])
        pcd_cropped.colors = o3d.utility.Vector3dVector(colors[mask])
        pcd_cropped.normals = o3d.utility.Vector3dVector(normals[mask])
        
        self.segments[segment_id] = pcd_cropped
        print(f"Cropped {segment_id}")
        self.visualize_current_state()
        return True

    def visualize_current_state(self):
        """Visualize current state of all segments"""
        if self.combined_group:
            segments_to_show = [self.segments[seg_id] for seg_id in self.combined_group]
            o3d.visualization.draw_geometries(segments_to_show)
        else:
            self.visualize_all()

pcd = o3d.io.read_point_cloud("/home/t/Desktop/work/results/combined_pointcloud1.pcd")

# Create the connector
connector = PointCloudConnector(pcd)

# Interactive connection loop
while True:
    print("\nOptions:")
    print("1. View all segments")
    print("2. View specific segment")
    print("3. Connect segments")
    print("4. View combined group")
    print("5. Translate segment")
    print("6. Rotate segment")
    print("7. Remove segment")
    print("8. Crop segment")
    print("9. Exit")
    
    choice = input("Enter your choice (1-9): ")

    
    if choice == '1':
        connector.visualize_all()
    
    elif choice == '2':
        segment_id = input("Enter segment ID (e.g., segment_0): ")
        connector.visualize_segment(segment_id)
    
    elif choice == '3':
        combined = connector.get_combined_group()
        if combined:
            print(f"\nCurrent combined group: {combined}")
            print("Add to combined group:")
            new_segment = input("Enter segment ID to add: ")
            last_segment = combined[-1]  # Connect to the last segment in the group
            connector.connect_and_visualize(last_segment, new_segment)
        else:
            print("\nStart new combination:")
            seg1 = input("Enter first segment ID: ")
            seg2 = input("Enter second segment ID: ")
            connector.connect_and_visualize(seg1, seg2)
    
    elif choice == '4':
        combined = connector.get_combined_group()
        if combined:
            print(f"Combined group: {combined}")
            combined_segments = [connector.segments[seg_id] for seg_id in combined]
            o3d.visualization.draw_geometries(combined_segments)
        else:
            print("No segments combined yet")
    
    elif choice == '5':
        segment_id = input("Enter segment ID to translate: ")
        try:
            x = float(input("Enter X translation: "))
            y = float(input("Enter Y translation: "))
            z = float(input("Enter Z translation: "))
            connector.translate_segment(segment_id, [x, y, z])
        except ValueError:
            print("Invalid input. Please enter numbers for translation.")

    elif choice == '6':
        segment_id = input("Enter segment ID to rotate: ")
        try:
            angle = float(input("Enter rotation angle in degrees: "))
            axis = input("Enter rotation axis (x/y/z): ").lower()
            if axis in ['x', 'y', 'z']:
                connector.rotate_segment(segment_id, angle, axis)
            else:
                print("Invalid axis. Please use x, y, or z.")
        except ValueError:
            print("Invalid input. Please enter a number for angle.")

    elif choice == '7':
        segment_id = input("Enter segment ID to remove: ")
        connector.remove_segment(segment_id)

    elif choice == '8':
        segment_id = input("Enter segment ID to crop: ")
        try:
            print("Enter minimum bounds (x,y,z):")
            min_x = float(input("Min X: "))
            min_y = float(input("Min Y: "))
            min_z = float(input("Min Z: "))
            print("Enter maximum bounds (x,y,z):")
            max_x = float(input("Max X: "))
            max_y = float(input("Max Y: "))
            max_z = float(input("Max Z: "))
            connector.crop_segment(segment_id, [min_x, min_y, min_z], [max_x, max_y, max_z])
        except ValueError:
            print("Invalid input. Please enter numbers for bounds.")

    elif choice == '9':
        break
    
    else:
        print("Invalid choice. Please try again.")