import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
import heapq 
from PIL import Image
import shutil

def load_reference_colors(folder_path):
    """
    Loads images from a folder and extracts a set of unique colors from each image.
    """
    unique_colors = set()

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        # Check if the file is an image file
        if os.path.isfile(filepath) and any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                # Extract unique colors from the image
                unique_colors.update({tuple(color) for row in img for color in row})

    return unique_colors

def highlight_differences(reference_image_path, target_image_path, unique_colors, output_path=None):
    """
    Highlights differences between the reference image and target image.
    Colors not found in the reference image are turned black in the target image.
    Then applies a median filter to smooth the image. Optionally saves the modified target image to 'output_path'.

    Args:
    - reference_image_path (str): Path to the reference image.
    - target_image_path (str): Path to the target image to process.
    - output_path (str, optional): If provided, the path to save the modified target image.
    - kernel_size (int): Size of the kernel for the median filter (must be an odd number).

    Returns:
    - The modified target image as a numpy array. If 'output_path' is provided, also saves the image.
    """
    kernel_size=5
    reference_colors = unique_colors
    if reference_colors is None:
        print("Reference image not found or the path is incorrect.")
        return None
    
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        print("Target image not found or the path is incorrect.")
        return None
    
    # Highlight differences by setting non-matching pixels to black
    for i in range(target_img.shape[0]):
        for j in range(target_img.shape[1]):
            pixel_color = tuple(target_img[i, j])
            if pixel_color not in reference_colors:
                target_img[i, j] = [0, 255, 0]
    
    # Apply a median filter to smooth the image
    if output_path:
        cv2.imwrite(output_path, target_img)
    
    return target_img


def highlight_colors_in_folder(reference_folder_path, target_folder_path, output_folder_path=None):
    """
    Processes each image in 'target_folder_path' comparing with 'reference_image_path',
    saves highlighted differences images to 'output_folder_path', and optionally replaces the original image.
    """
    unique_colors=load_reference_colors(reference_folder_path)
    for filename in os.listdir(target_folder_path):
        if filename.lower().endswith('.png'):
            target_image_path = os.path.join(target_folder_path, filename)
            #print(f"Processing {filename}...")
            processed_image = highlight_differences(reference_folder_path, target_image_path, unique_colors)
            
            if processed_image is not None:
                # Save a copy in the output folder if specified
                if output_folder_path:
                    if not os.path.exists(output_folder_path):
                        os.makedirs(output_folder_path)
                    output_image_path = os.path.join(output_folder_path, filename)
                    cv2.imwrite(output_image_path, processed_image)
                
                # Replace the original image
                cv2.imwrite(target_image_path, processed_image)



def divide_image_into_grids(base_dir, output_dir_name='patches', initial_grid_size=8):

    """
    Divides all images in a specified directory into smaller grid patches and saves them in a subdirectory.

    This function takes each image in the provided directory, divides it into a grid of smaller patches, and saves those patches into a specified subdirectory. The grid size is initially set but may be adjusted to ensure each patch is of equal size and the original image dimensions are divisible by the grid size. This is particularly useful for processing images in machine learning tasks where uniform input sizes are needed.

    Parameters:
    - base_dir (str): The path to the directory containing the original images.
    - output_dir_name (str, optional): The name of the subdirectory where the image patches will be saved. Defaults to 'patches'.
    - initial_grid_size (int, optional): The initial size of the grid (e.g., 8 for an 8x8 grid). The function may adjust this size to fit the image dimensions. Defaults to 8.

    Returns:
    - None: The function saves the image patches in the specified subdirectory and does not return any value.
    """
    print(f"Dividing image in {base_dir} into subgrids")
    # Attempt to get the size of the first image in the directory
    for filename in os.listdir(base_dir):
        if filename.endswith('.png'):
            first_image_path = os.path.join(base_dir, filename)
            with Image.open(first_image_path) as img:
                img_size = img.size[0]  # Assuming the images are square
                break
    
    # Adjust the grid size if the image size is not divisible by the initial grid size
    while img_size % initial_grid_size != 0 or (img_size // initial_grid_size) % 2 != 0:
        initial_grid_size -= 1  # Decrement grid_size
    if initial_grid_size < 1:
        raise ValueError("Cannot find a suitable grid size.")  # Decrement grid_size until img_size is divisible by grid_size
    grid_size = initial_grid_size
    patch_size = img_size // grid_size

    
    # Create a patches directory within the base directory if it does not exist
    patches_dir = os.path.join(base_dir, output_dir_name)
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)
    #print(f"Patches will be saved in: {patches_dir}, grid size: {grid_size}, patch size: {patch_size}x{patch_size}")
    
    # Process each image in the base directory
    for filename in os.listdir(base_dir):
        if filename.endswith('.png'):  # Ensure we're only processing images
            image_path = os.path.join(base_dir, filename)
            with Image.open(image_path) as img:
                # Extract patches
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Define the bounding box for each patch
                        box = (i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size)
                        # Extract the patch
                        patch = img.crop(box)
                        # Construct the patch filename
                        patch_filename = f'{filename[:-4]}x{i}x{j}.png'
                        patch_path = os.path.join(patches_dir, patch_filename)
                        # Save the patch
                        patch.save(patch_path)



def flatten_and_prefix_files(base_dir):
    """
    Flattens the directory structure by moving all files from subdirectories to the base directory and prefixes the filenames with the subdirectory name.

    This function iterates through each subdirectory in the specified base directory, moves each file to the base directory, and prefixes the filename with the name of its original subdirectory. This operation is useful for consolidating files from multiple subdirectories into a single directory and ensuring their names remain unique.

    Parameters:
    - base_dir (str): The path to the base directory containing the subdirectories and files to be flattened and prefixed.

    Returns:
    - None: The function renames and moves files but does not return any value.
    """

    # Iterate over each subdirectory in the base directory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        # Only proceed if it's a directory and not a file
        if os.path.isdir(subdir_path):
            # Process each file in the subdirectory
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                # Only proceed if it's a file and not a subdirectory
                if os.path.isfile(file_path):
                    # Prefix the filename with the subdirectory name
                    new_filename = f"{subdir}_{filename}"
                    new_file_path = os.path.join(base_dir, new_filename)
                    # Move the file to the base directory
                    shutil.move(file_path, new_file_path)
                    print(f"Moved and renamed file to: {new_file_path}")
            
            # After processing all files, remove the now-empty subdirectory
            os.rmdir(subdir_path)
            print(f"Removed empty directory: {subdir_path}")

def get_anomaly_type(image_filename):
    """
    Extracts the anomaly type from the image filename.
    """
    # Assuming the anomaly type can contain underscores and is followed by a set of indices separated by 'x'
    # We join all parts before the first occurrence of 'x'
    parts = image_filename.split('x')
    anomaly_type = '_'.join(parts[0].split('_')[:-1])
    return anomaly_type

