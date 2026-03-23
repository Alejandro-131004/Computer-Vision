import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, clear_output
import os
import random
import pandas as pd
import glob

# Helper to show images side by side
def show(imgs, titles=None, figsize=(16, 5)):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def order_corners(pts):
    """
    Order 4 corner points as: TopLeft, TopRight, BottomRight, BottomLeft:
        1. Sort by y to get top vs bottom 
        2. Sort by x inside each pair
    """
    sorted_by_y = pts[np.argsort(pts[:, 1])]

    top = sorted_by_y[:2]
    top = top[np.argsort(top[:, 0])]  # left first
    bottom = sorted_by_y[2:]
    bottom = bottom[np.argsort(bottom[:, 0])]  # left first

    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

def annotate_image_dataset(image_folder):
    """
    Iterates through images in a folder, displays them, 
    and prompts the user for the number of balls.
    Returns a pandas DataFrame with the filenames and counts.
    """
    annotations = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    # Check if directory exists
    if not os.path.exists(image_folder):
        print(f"Error: Directory '{image_folder}' not found.")
        return pd.DataFrame()

    # Filter and sort only image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print(f"No images found in {image_folder}.")
        return pd.DataFrame()

    for filename in image_files:
        img_path = os.path.join(image_folder, filename)
        
        # Read and show the image
        img = mpimg.imread(img_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"File: {filename}")
        plt.show()
        
        # Keep asking until a valid integer or 'q' is provided
        while True:
            try:
                user_input = input(f"Enter the number of balls for '{filename}' (or type 'q' to quit): ")
                
                if user_input.lower() == 'q':
                    print("Manual annotation interrupted.")
                    clear_output(wait=True)
                    return pd.DataFrame(annotations, columns=['filename', 'ball_count'])
                
                num_balls = int(user_input)
                
                if num_balls < 0:
                    print("Error: The number of balls cannot be negative.")
                    continue
                
                break # Valid input, exit while loop
                
            except ValueError:
                print("Error: Invalid input. Please enter an integer.")
        
        # Store the result
        annotations.append({'filename': filename, 'ball_count': num_balls})
        
        # Clear the cell output to simulate the "game" transition
        clear_output(wait=True)
        
    print("Annotation process finished successfully!")
    
    # Create and return the DataFrame
    df_results = pd.DataFrame(annotations, columns=['filename', 'ball_count'])
    return df_results