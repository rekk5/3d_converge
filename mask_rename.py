import os

def rename_to_frame_style(directory, ext):
    for filename in sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0])):
        name, _ = os.path.splitext(filename)
        new_name = f"frame_{int(name):04d}{ext}"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

base_path = "/home/t/OnlineAnySeg//data/My_dataset/seg_result/My_sequence1"
rename_to_frame_style(os.path.join(base_path, "mask"), ".png")
rename_to_frame_style(os.path.join(base_path, "mask_color"), ".png")
rename_to_frame_style(os.path.join(base_path, "mask_embeddings"), ".pt")
