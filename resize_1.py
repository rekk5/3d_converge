from PIL import Image
from pathlib import Path

input_dir = Path("/home/t/OnlineAnySeg/data/My_dataset/My_sequence1/depth")
output_dir = Path("/home/t/OnlineAnySeg/data/My_dataset/My_sequence1/depth_resized_320x240")
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    img = Image.open(img_path).resize((320, 240))
    img.save(output_dir / img_path.name)