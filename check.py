import cv2
import numpy as np
import matplotlib.pyplot as plt
depth = cv2.imread("/home/t/OnlineAnySeg/data/My_dataset/seg_result/My_sequence1/mask/0040.png", cv2.IMREAD_UNCHANGED)
print("Depth shape:", depth.shape)
print("Unique values:", np.unique(depth))
print("Min:", depth.min(), "Max:", depth.max())
print(depth.dtype, depth.min(), depth.max())
depth_vis = np.clip(depth, 1, 8000)  # ignore zeros
plt.imshow(depth_vis, cmap='plasma')
plt.colorbar()
plt.show()
