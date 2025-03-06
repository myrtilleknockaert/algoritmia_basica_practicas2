#!/usr/bin/env python3
"""
Seam Carving Implementation

Usage:
    ./seam_carving.py <num_seams> <image_file> <output_directory>
    
Example:
    ./seam_carving.py 50 experimentacion/elefante.jpg ./
"""

import sys
import os
import time  # Module to measure wall and CPU time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
from typing import List, Tuple


def color_sum(img: np.ndarray) -> np.ndarray:
    """
    Create a matrix by adding a black border around the image and
    computing the sum of the RGB values for each pixel.

    :param img: Input image as a numpy array of shape (height, width, channels)
    :return: New matrix with a 1-pixel black border (shape: (height+2, width+2))
    """
    height, width, _ = img.shape
    # Initialize a new matrix with an extra border filled with zeros (black)
    new_matrix = np.zeros((height + 2, width + 2))
    # Loop over each pixel in the original image
    for i in range(height):
        for j in range(width):
            # Place the sum of the RGB values in the corresponding position,
            # offset by one in both dimensions due to the added border.
            new_matrix[i + 1, j + 1] = np.sum(img[i, j])
    return new_matrix


def energy(matrix: np.ndarray, i: int, j: int) -> float:
    """
    Compute the energy of the pixel at (i, j) using the defined formula.
    A black border (value 0) is assumed outside the image.

    :param matrix: Matrix from color_sum with a black border.
    :param i: Row index.
    :param j: Column index.
    :return: Energy value.
    """
    # Get brightness values from the 8 neighbors (using 0 for missing neighbors)
    a = matrix[i - 1, j - 1] if i > 0 and j > 0 else 0
    b = matrix[i - 1, j] if i > 0 else 0
    c = matrix[i - 1, j + 1] if i > 0 and j < matrix.shape[1] - 1 else 0
    d = matrix[i, j - 1] if j > 0 else 0
    f = matrix[i, j + 1] if j < matrix.shape[1] - 1 else 0
    g = matrix[i + 1, j - 1] if i < matrix.shape[0] - 1 and j > 0 else 0
    h = matrix[i + 1, j] if i < matrix.shape[0] - 1 else 0
    ii = (
        matrix[i + 1, j + 1]
        if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1
        else 0
    )

    # Compute differences in horizontal and vertical directions
    x = a + 2 * d + g - c - 2 * f - ii
    y = a + 2 * b + c - g - 2 * h - ii
    # Return the Euclidean norm of the differences (the energy)
    return sqrt(x**2 + y**2)


def energy_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the energy for each pixel (excluding the border).

    :param matrix: Matrix from color_sum.
    :return: 2D numpy array of energy values.
    """
    height = matrix.shape[0] - 2  # Remove border rows
    width = matrix.shape[1] - 2  # Remove border columns
    new_matrix = np.zeros((height, width))
    # Compute energy for each pixel inside the original image area
    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            new_matrix[i - 1, j - 1] = energy(matrix, i, j)
    return new_matrix


def find_path(energy_mat: np.ndarray) -> List[List]:
    """
    Compute the cumulative energy (cost) matrix and record the direction for backtracking.

    Each element in the returned matrix is a pair [cumulative_cost, direction] where
    direction is -1, 0, or 1 indicating the previous column offset.

    :param energy_mat: 2D numpy array of energy values.
    :return: Cumulative cost matrix as a list of lists.
    """
    # Convert the energy matrix to a list of lists for easier manipulation
    path_matrix = energy_mat.tolist()
    height = len(path_matrix)
    width = len(path_matrix[0])

    # Initialize the first row: cost equals energy, with a dummy direction 0
    for j in range(width):
        path_matrix[0][j] = [path_matrix[0][j], 0]

    # Use dynamic programming to fill the rest of the path matrix
    for i in range(1, height):
        # Process first column separately
        prev_mid = path_matrix[i - 1][0][0]
        prev_right = path_matrix[i - 1][1][0] if width > 1 else float("inf")
        if prev_mid <= prev_right:
            path_matrix[i][0] = [prev_mid + path_matrix[i][0], 0]
        else:
            path_matrix[i][0] = [prev_right + path_matrix[i][0], 1]

        # Process middle columns
        for j in range(1, width - 1):
            left_cost = path_matrix[i - 1][j - 1][0]
            mid_cost = path_matrix[i - 1][j][0]
            right_cost = path_matrix[i - 1][j + 1][0]
            min_cost = min(left_cost, mid_cost, right_cost)
            # Set direction based on which neighbor provided the minimum cost
            if min_cost == left_cost:
                direction = -1
            elif min_cost == mid_cost:
                direction = 0
            else:
                direction = 1
            path_matrix[i][j] = [min_cost + path_matrix[i][j], direction]

        # Process last column separately
        if width > 1:
            left_cost = path_matrix[i - 1][width - 2][0]
            mid_cost = path_matrix[i - 1][width - 1][0]
            if left_cost <= mid_cost:
                path_matrix[i][width - 1] = [left_cost + path_matrix[i][width - 1], -1]
            else:
                path_matrix[i][width - 1] = [mid_cost + path_matrix[i][width - 1], 0]

    return path_matrix


def min_in_list(lst: List[List]) -> Tuple[List, int]:
    """
    Find the element with the minimum cumulative cost in a list of [cost, direction] pairs.

    :param lst: List of pairs.
    :return: Tuple of the pair and its index.
    """
    min_val = lst[0][0]
    pos = 0
    for i in range(len(lst)):
        if lst[i][0] < min_val:
            min_val = lst[i][0]
            pos = i
    return lst[pos], pos


def retrieve_seam(path_matrix: List[List]) -> List[int]:
    """
    Backtrack through the cumulative cost matrix to retrieve the seam.

    :param path_matrix: Matrix computed by find_path.
    :return: List of column indices (one per row) representing the seam.
    """
    seam = []
    # Find the minimum cost in the bottom row and its column index
    _, pos = min_in_list(path_matrix[-1])
    height = len(path_matrix)
    # Backtrack from bottom to top using the stored direction
    for i in range(height - 1, -1, -1):
        seam.append(pos)
        _, direction = path_matrix[i][pos]
        pos += direction  # Move to the corresponding column in the previous row
    seam.reverse()  # Reverse to have the seam from top to bottom
    return seam


def remove_seam(image: np.ndarray, seam: List[int]) -> np.ndarray:
    """
    Remove the specified seam from the image.

    :param image: Original image as a numpy array.
    :param seam: List of column indices to remove (one per row).
    :return: New image with one less column.
    """
    h, w, c = image.shape
    # Initialize a new image with one fewer column
    new_image = np.zeros((h, w - 1, c), dtype=image.dtype)
    # Loop through each row and delete the pixel at the specified column
    for i in range(h):
        new_image[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)
    return new_image


def process_seam(image: np.ndarray) -> np.ndarray:
    """
    Process the image by removing one seam and print dimensions before and after removal.

    :param image: Input image as a numpy array.
    :return: New image with one seam removed.
    """
    # Print the dimensions before seam removal
    print("Image dimensions:", image.shape)
    # Calculate the color sum with border
    cs = color_sum(image)
    # Calculate the energy matrix
    em = energy_matrix(cs)
    # Compute the cumulative cost matrix and directions
    path_mat = find_path(em)
    # Retrieve the optimal seam based on the cumulative cost
    seam = retrieve_seam(path_mat)
    # Remove the seam from the image
    new_img = remove_seam(image, seam)
    # Print the dimensions after seam removal
    print("Image dimensions after seam removal:", new_img.shape)
    return new_img


def final_function(num_seams: int, image_file: str, output_dir: str) -> None:
    """
    Iteratively remove seams from the image and save only the final result.

    :param num_seams: Number of seams (columns) to remove.
    :param image_file: Path to the input image.
    :param output_dir: Directory to save the final result.
    """
    # Read the original image from file
    img = plt.imread(image_file)
    # Iteratively process the image, removing one seam at a time
    for i in range(num_seams):
        print("Iteration:", i)
        img = process_seam(img)
    # Build the final output file path
    final_output_path = os.path.join(output_dir, "final_result.png")
    # Print the final image dimensions
    print("Final image dimensions:", img.shape)
    # Save the final image result
    mpimg.imsave(final_output_path, img)
    print("Saved final result image:", final_output_path)

if __name__ == "__main__":
    # Ensure the proper number of command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: costuras <num_seams> <image_file> <output_directory>")
        sys.exit(1)
    # Convert the number of seams argument to an integer
    try:
        num_seams = int(sys.argv[1])
    except ValueError:
        print("Error: <num_seams> must be an integer.")
        sys.exit(1)

    image_file = sys.argv[2]
    output_dir = sys.argv[3]

    # Check if the input image file exists
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' does not exist.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Record start times for wall-clock and CPU
    start_wall = time.time()
    start_cpu = os.times()

    # Execute the final function to remove seams and save the final image
    final_function(num_seams, image_file, output_dir)

    # Record end times for wall-clock and CPU
    end_wall = time.time()
    end_cpu = os.times()

    # Calculate total elapsed time, user CPU time, and system CPU time
    total_time = end_wall - start_wall
    user_time = end_cpu.user - start_cpu.user
    system_time = end_cpu.system - start_cpu.system
    # Compute CPU utilization percentage
    cpu_utilization = (
        (user_time + system_time) / total_time * 100 if total_time > 0 else 0
    )

    # Display a clear summary of the execution time in English
    print(
        f"\nThe script completed in a total of {total_time:.3f} seconds, "
        f"using {user_time:.2f} seconds of user time and {system_time:.2f} seconds of system time, "
        f"with an average CPU utilization of {cpu_utilization:.0f}%."
    )
