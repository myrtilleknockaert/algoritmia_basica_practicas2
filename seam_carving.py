import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
from typing import List, Tuple

# Load and display the image
img = plt.imread("photo.jpg")
plt.imshow(img)
plt.axis("off")  # Remove axes
plt.show()


def color_sum(img: np.ndarray) -> np.ndarray:
    """
    Create a matrix by adding a black border around the image and computing
    the sum of the RGB values for each pixel.

    :param img: Input image as a numpy array of shape (height, width, channels)
    :return: New matrix with a black border (shape: (height+2, width+2))
    """
    height, width, _ = img.shape
    new_matrix = np.zeros((height + 2, width + 2))  # 1-pixel black border (value 0)

    # Fill the matrix with the sum of RGB values for each pixel
    for i in range(height):
        for j in range(width):
            new_matrix[i + 1, j + 1] = np.sum(img[i, j])

    return new_matrix


def energy(matrix: np.ndarray, i: int, j: int) -> float:
    """
    Compute the energy of the pixel at position (i, j) using the given formula.

    The energy is defined as:
        energy = sqrt(energyX^2 + energyY^2)
    where:
        energyX = a + 2*d + g - c - 2*f - ii
        energyY = a + 2*b + c - g - 2*h - ii
    with a,b,c,d,e,f,g,h,ii corresponding to the brightness of the neighboring pixels.

    A black border (value 0) is assumed outside the image.

    :param matrix: Matrix obtained from color_sum with an added border.
    :param i: Row index (must be within 1..matrix.shape[0]-2)
    :param j: Column index (must be within 1..matrix.shape[1]-2)
    :return: Energy value as a float.
    """
    a = matrix[i - 1, j - 1] if i > 0 and j > 0 else 0
    b = matrix[i - 1, j] if i > 0 else 0
    c = matrix[i - 1, j + 1] if i > 0 and j < matrix.shape[1] - 1 else 0
    d = matrix[i, j - 1] if j > 0 else 0
    e = matrix[i, j]
    f = matrix[i, j + 1] if j < matrix.shape[1] - 1 else 0
    g = matrix[i + 1, j - 1] if i < matrix.shape[0] - 1 and j > 0 else 0
    h = matrix[i + 1, j] if i < matrix.shape[0] - 1 else 0
    # Rename the bottom-right neighbor to avoid conflict with index i
    ii = (
        matrix[i + 1, j + 1]
        if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1
        else 0
    )

    x = a + 2 * d + g - c - 2 * f - ii
    y = a + 2 * b + c - g - 2 * h - ii

    return sqrt(x**2 + y**2)


def energy_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the energy for each pixel (excluding the border) and return a new matrix.

    :param matrix: Matrix obtained from color_sum.
    :return: A 2D numpy array with energy values.
    """
    height, width = matrix.shape[0] - 2, matrix.shape[1] - 2
    new_matrix = np.zeros((height, width))

    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            new_matrix[i - 1, j - 1] = energy(matrix, i, j)

    return new_matrix


def find_path(matrix: np.ndarray) -> List[List]:
    """
    Compute the dynamic programming matrix for seam carving.

    Each element in the returned matrix is a pair [cumulative_cost, direction],
    where 'direction' indicates the relative position (-1, 0, or 1) from the previous row.

    :param matrix: 2D numpy array of energy values.
    :return: A list of lists (converted from the matrix) with cost and path direction.
    """
    # Convert the energy matrix to a list of lists for easier manipulation
    path_matrix = matrix.tolist()
    height = len(path_matrix)
    width = len(path_matrix[0])

    # Initialize the first row with cost and a dummy direction 0
    for j in range(width):
        path_matrix[0][j] = [path_matrix[0][j], 0]

    # Dynamic programming: fill in the path_matrix with cumulative cost and direction.
    for i in range(1, height):
        # First column
        prev_mid = path_matrix[i - 1][0][0]
        prev_right = path_matrix[i - 1][1][0] if width > 1 else float("inf")
        if prev_mid < prev_right:
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
        prev_left = path_matrix[i - 1][width - 2][0]
        prev_mid = path_matrix[i - 1][width - 1][0]
        if prev_mid < prev_left:
            path_matrix[i][width - 1] = [prev_mid + path_matrix[i][width - 1], 0]
        else:
            path_matrix[i][width - 1] = [prev_left + path_matrix[i][width - 1], -1]

    return path_matrix


def min_in_list(lst: List[List]) -> Tuple[List, int]:
    """
    Find the element with the minimum cost in a list of [cost, direction] pairs.

    :param lst: List of pairs.
    :return: A tuple containing the element [cost, direction] and its index.
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
    Backtrack through the dynamic programming matrix to retrieve the seam (column indices).

    :param path_matrix: The computed matrix with cumulative costs and directions.
    :return: List of column indices representing the seam from top to bottom.
    """
    seam = []
    _, pos = min_in_list(path_matrix[-1])

    # Backtrack from the bottom row up to the top row
    for i in range(len(path_matrix) - 1, -1, -1):
        seam.append(pos)
        # The second element is the direction to move in the previous row
        _, direction = path_matrix[i][pos]
        pos += direction

    seam.reverse()  # Optionally reverse to have the seam from top to bottom
    return seam


def remove_seam(image: np.ndarray, seam: List[int]) -> np.ndarray:
    """
    Remove the specified seam from the image.

    :param image: Original image as a numpy array.
    :param seam: List of column indices (one per row) to remove.
    :return: New image with one less column.
    """
    h, w, c = image.shape
    new_image = np.zeros((h, w - 1, c), dtype=image.dtype)

    for i in range(h):
        new_image[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)

    return new_image


def semi_final_function(photo: str, directory: str, iteration: int) -> str:
    """
    Process the image by removing one seam and save the result.

    :param photo: Path to the input image file.
    :param directory: Directory where the result image will be saved.
    :param iteration: Current iteration number (used for naming the output file).
    :return: Path to the saved output image.
    """
    img = plt.imread(photo)
    print("Image dimensions:", img.shape)

    # Compute the energy matrix using the sum of colors and energy function.
    color_matrix = color_sum(img)
    energy_mat = energy_matrix(color_matrix)

    # Compute the cumulative cost matrix for seam carving.
    path_matrix = find_path(energy_mat)

    # Retrieve the seam (path) with the minimum cumulative energy.
    seam = retrieve_seam(path_matrix)

    # Remove the seam from the original image.
    new_img = remove_seam(img, seam)
    plt.imshow(new_img)
    plt.axis("off")
    plt.show()

    output_path = f"{directory}result{iteration}.png"
    print("Image dimensions after seam removal:", new_img.shape)
    mpimg.imsave(output_path, new_img)

    return output_path


def final_function(num_seams: int, image_file: str, directory: str) -> None:
    """
    Iteratively remove a specified number of seams from the image.

    :param num_seams: Number of seams to remove.
    :param image_file: Path to the input image file.
    :param directory: Directory where result images will be saved.
    """
    for i in range(num_seams):
        print("Iteration:", i)
        image_file = semi_final_function(image_file, directory, i)


# Example usage (à adapter selon vos besoins) :
# final_function(5, "photo.png", "/tmp/")
