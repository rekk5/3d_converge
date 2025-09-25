import os
from pathlib import Path
import re

def numeric_sort_key(file_path):
    # Extract numeric part from filename, e.g., "12.png" -> 12
    match = re.search(r'(\d+)', file_path.stem)
    return int(match.group(1)) if match else -1

def renumber_files_with_padding(folder, ext, digits=4):
    folder = Path(folder)
    files = sorted(folder.glob(f'*.{ext}'), key=numeric_sort_key)

    for new_idx, old_file in enumerate(files):
        new_name = f"{new_idx:0{digits}d}.{ext}"
        new_file = folder / new_name
        print(f"Renaming {old_file.name} -> {new_file.name}")
        old_file.rename(new_file)

# Base directory
base_path = Path('/home/t/OnlineAnySeg/data/My_dataset/My_sequence1')

# Renaming files with zero-padded names
renumber_files_with_padding(base_path / 'color', 'jpg', digits=4)
renumber_files_with_padding(base_path / 'depth', 'png', digits=4)
renumber_files_with_padding(base_path / 'poses', 'txt', digits=4)

