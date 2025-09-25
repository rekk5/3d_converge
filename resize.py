import cv2
import os
from pathlib import Path

def resize_and_rename_images(folder, ext, target_size=(640, 480)):
    folder = Path(folder)
    files = sorted(folder.glob(f'*.{ext}'), key=lambda x: int(x.stem))
    for idx, img_path in enumerate(files):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST if ext == "png" else cv2.INTER_AREA)
        out_path = folder / f"{str(idx).zfill(4)}.{ext}"
        print(f"Saving resized: {img_path.name} -> {out_path.name}")
        cv2.imwrite(str(out_path), resized)

# Run on your dataset
base_path = Path("/home/t/OnlineAnySeg/data/My_dataset/My_sequence1")
resize_and_rename_images(base_path / "color", "jpg")  # RGB images
resize_and_rename_images(base_path / "depth", "png")  # Depth images