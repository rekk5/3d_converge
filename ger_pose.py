import os
import numpy as np

# Path to your input pose file (the one you pasted)
input_pose_file = '/home/t/Desktop/labger/poses_odom.txt'  # change if different
output_pose_dir = '/home/t/Desktop/oaseg_ger/My_dataset/My_sequence1/poses'

os.makedirs(output_pose_dir, exist_ok=True)

with open(input_pose_file, 'r') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    vals = list(map(float, line.strip().split()))
    if len(vals) != 12:
        print(f"Skipping line {idx}: doesn't have 12 values")
        continue
    mat = np.array(vals).reshape(3, 4)
    bottom_row = np.array([[0, 0, 0, 1]])
    mat4x4 = np.vstack((mat, bottom_row))
    
    output_path = os.path.join(output_pose_dir, f"{idx}.txt")
    np.savetxt(output_path, mat4x4, fmt='%.6f')