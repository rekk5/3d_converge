import pye57
import numpy as np
from plyfile import PlyData, PlyElement

def e57_to_ply(input_file, output_file):
    """
    Convert E57 file to PLY format
    
    Args:
        input_file (str): Path to input E57 file
        output_file (str): Path to output PLY file
    """
    # Read the E57 file
    e57 = pye57.E57(input_file)
    
    # Get the first scan using the correct method syntax and ignore missing fields
    data_raw = e57.read_scan(0, ignore_missing_fields=True)
    
    # Create a dictionary with the required data
    data = {
        'cartesianX': data_raw['cartesianX'],
        'cartesianY': data_raw['cartesianY'],
        'cartesianZ': data_raw['cartesianZ'],
        'intensity': data_raw['intensity'] if 'intensity' in data_raw else np.zeros_like(data_raw['cartesianX'])
    }
    
    # Create structured array for PLY
    vertices = np.zeros(len(data['cartesianX']), dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('intensity', 'f4')
    ])
    
    # Fill the array with data
    vertices['x'] = data['cartesianX']
    vertices['y'] = data['cartesianY']
    vertices['z'] = data['cartesianZ']
    vertices['intensity'] = data['intensity']
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertices, 'vertex')
    
    # Create and save PLY file
    PlyData([vertex_element], text=False).write(output_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python e57_to_ply.py input.e57 output.ply")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    e57_to_ply(input_file, output_file)
    print(f"Successfully converted {input_file} to {output_file}")