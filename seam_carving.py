#!/usr/bin/env python3
"""
Seam Carving Implementation

Usage:
    costuras <num_seams> <image_file> <output_directory>
    
Example:
    costuras 5 photo.png /tmp/
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
from typing import List, Tuple

def color_sum(img: np.ndarray) -> np.ndarray:
    height, width, _ = img.shape
    new_matrix = np.zeros((height + 2, width + 2))
    for i in range(height):
        for j in range(width):
            new_matrix[i + 1, j + 1] = np.sum(img[i, j])
    return new_matrix

def energy(matrix: np.ndarray, i: int, j: int) -> float:
    a = matrix[i - 1, j - 1] if i > 0 and j > 0 else 0
    b = matrix[i - 1, j] if i > 0 else 0
    c = matrix[i - 1, j + 1] if i > 0 and j < matrix.shape[1] - 1 else 0
    d = matrix[i, j - 1] if j > 0 else 0
    f = matrix[i, j + 1] if j < matrix.shape[1] - 1 else 0
    g = matrix[i + 1, j - 1] if i < matrix.shape[0] - 1 and j > 0 else 0
    h = matrix[i + 1, j] if i < matrix.shape[0] - 1 else 0
    ii = matrix[i + 1, j + 1] if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1 else 0

    x = a + 2 * d + g - c - 2 * f - ii
    y = a + 2 * b + c - g - 2 * h - ii
    return sqrt(x ** 2 + y ** 2)

def energy_matrix(matrix: np.ndarray) -> np.ndarray:
    height = matrix.shape[0] - 2
    width = matrix.shape[1] - 2
    new_matrix = np.zeros((height, width))
    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            new_matrix[i - 1, j - 1] = energy(matrix, i, j)
    return new_matrix

def find_path(energy_mat: np.ndarray) -> List[List]:
    path_matrix = energy_mat.tolist()
    height = len(path_matrix)
    width = len(path_matrix[0])
    for j in range(width):
        path_matrix[0][j] = [path_matrix[0][j], 0]
    for i in range(1, height):
        # First column
        prev_mid = path_matrix[i - 1][0][0]
        prev_right = path_matrix[i - 1][1][0] if width > 1 else float('inf')
        if prev_mid <= prev_right:
            path_matrix[i][0] = [prev_mid + path_matrix[i][0], 0]
        else:
            path_matrix[i][0] = [prev_right + path_matrix[i][0], 1]
        # Middle columns
        for j in range(1, width - 1):
            left_cost = path_matrix[i - 1][j - 1][0]
            mid_cost = path_matrix[i - 1][j][0]
            right_cost = path_matrix[i - 1][j + 1][0]
            min_cost = min(left_cost, mid_cost, right_cost)
            if min_cost == left_cost:
                direction = -1
            elif min_cost == mid_cost:
                direction = 0
            else:
                direction = 1
            path_matrix[i][j] = [min_cost + path_matrix[i][j], direction]
        # Last column
        if width > 1:
            left_cost = path_matrix[i - 1][width - 2][0]
            mid_cost = path_matrix[i - 1][width - 1][0]
            if left_cost <= mid_cost:
                path_matrix[i][width - 1] = [left_cost + path_matrix[i][width - 1], -1]
            else:
                path_matrix[i][width - 1] = [mid_cost + path_matrix[i][width - 1], 0]
    return path_matrix

def min_in_list(lst: List[List]) -> Tuple[List, int]:
    min_val = lst[0][0]
    pos = 0
    for i in range(len(lst)):
        if lst[i][0] < min_val:
            min_val = lst[i][0]
            pos = i
    return lst[pos], pos

def retrieve_seam(path_matrix: List[List]) -> List[int]:
    seam = []
    _, pos = min_in_list(path_matrix[-1])
    height = len(path_matrix)
    for i in range(height - 1, -1, -1):
        seam.append(pos)
        _, direction = path_matrix[i][pos]
        pos += direction
    seam.reverse()
    return seam

def remove_seam(image: np.ndarray, seam: List[int]) -> np.ndarray:
    h, w, c = image.shape
    new_image = np.zeros((h, w - 1, c), dtype=image.dtype)
    for i in range(h):
        new_image[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)
    return new_image

def process_seam(image: np.ndarray) -> np.ndarray:
    print("Image dimensions:", image.shape)
    cs = color_sum(image)
    em = energy_matrix(cs)
    path_mat = find_path(em)
    seam = retrieve_seam(path_mat)
    new_img = remove_seam(image, seam)
    print("Image dimensions after seam removal:", new_img.shape)
    return new_img

def final_function(num_seams: int, image_file: str, output_dir: str) -> None:
    img = plt.imread(image_file)
    for i in range(num_seams):
        print("Iteration:", i)
        img = process_seam(img)
    final_output_path = os.path.join(output_dir, "final_result.png")
    print("Final image dimensions:", img.shape)
    mpimg.imsave(final_output_path, img)
    print("Saved final result image:", final_output_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: costuras <num_seams> <image_file> <output_directory>")
        sys.exit(1)
    try:
        num_seams = int(sys.argv[1])
    except ValueError:
        print("Error: <num_seams> must be an integer.")
        sys.exit(1)
    
    image_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Mesure du temps d'exécution
    start_wall = time.time()
    start_cpu = os.times()
    
    final_function(num_seams, image_file, output_dir)
    
    end_wall = time.time()
    end_cpu = os.times()
    
    total_time = end_wall - start_wall
    user_time = end_cpu.user - start_cpu.user
    system_time = end_cpu.system - start_cpu.system
    cpu_utilization = (user_time + system_time) / total_time * 100 if total_time > 0 else 0
    
    print(f"\nThe script completed in a total of {total_time:.3f} seconds, "
          f"using {user_time:.2f} seconds of user time and {system_time:.2f} seconds of system time, "
          f"with an average CPU utilization of {cpu_utilization:.0f}%.")
