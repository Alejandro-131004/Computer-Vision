import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
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
    
