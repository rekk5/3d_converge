from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size=(512, 384)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(input_folder, filename)
            try:
                with Image.open(file_path) as img:
                    # Resize image to the new dimensions
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    # Save the resized image in the output folder with the same filename
                    output_path = os.path.join(output_folder, filename)
                    resized_img.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    # Specify the input folder containing the images and the desired output folder
    input_folder = "/home/t/Desktop/lab2"
    output_folder = "/home/t/Desktop/lab2rend"
    resize_images(input_folder, output_folder)
