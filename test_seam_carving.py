import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from seam_carving import (
    color_sum,
    energy,
    energy_matrix,
    find_path,
    retrieve_seam,
    remove_seam,
)


def test_color_sum() -> None:
    """
    Test color_sum by creating a small dummy image and verifying that the black border
    is correctly added and that the inner values correspondent to the sum of the RGB channels.
    """
    # Dummy image 2x2 with 3 channels
    dummy_img = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8
    )
    result = color_sum(dummy_img)
    expected = np.array(
        [[0, 0, 0, 0], [0, 60, 150, 0], [0, 240, 330, 0], [0, 0, 0, 0]], dtype=float
    )
    assert np.allclose(result, expected), "color_sum failed"
    print("test_color_sum passed")


def test_energy() -> None:
    """
    Test the energy function with a simple padded matrix.
    """
    # Create a 3x3 dummy matrix with a single nonzero value in the center
    dummy_matrix = np.array([[0, 0, 0], [0, 60, 0], [0, 0, 0]], dtype=float)
    # Add a border to simulate the black frame
    padded_matrix = np.pad(
        dummy_matrix, ((1, 1), (1, 1)), mode="constant", constant_values=0
    )
    e_val = energy(padded_matrix, 1, 1)
    print("Energy at center:", e_val)
    assert e_val >= 0, "energy should be non-negative"
    print("test_energy passed")


def test_energy_matrix() -> None:
    """
    Test energy_matrix by verifying the shape of the computed energy matrix.
    """
    dummy_img = np.array(
        [[[10, 10, 10], [20, 20, 20]], [[30, 30, 30], [40, 40, 40]]], dtype=np.uint8
    )
    cs = color_sum(dummy_img)
    em = energy_matrix(cs)
    # The dummy image 2x2 should produce an energy matrix of shape (2, 2)
    assert em.shape == (2, 2), "energy_matrix shape incorrect"
    print("test_energy_matrix passed")


def test_find_path_and_retrieve_seam() -> None:
    """
    Test the dynamic programming path calculation and seam retrieval on a dummy energy matrix.
    """
    dummy_energy = np.array([[10, 50, 10], [20, 5, 20], [30, 10, 30]], dtype=float)
    path_mat = find_path(dummy_energy)
    seam = retrieve_seam(path_mat)
    # The seam should have one index par row (3 rows)
    assert len(seam) == 3, "seam length incorrect"
    print("Seam found:", seam)
    print("test_find_path_and_retrieve_seam passed")


def test_remove_seam() -> None:
    """
    Test remove_seam by creating a dummy image and removing a known seam.
    """
    dummy_img = np.arange(27).reshape((3, 3, 3)).astype(np.uint8)
    seam = [1, 1, 1]  # Remove the middle column
    new_img = remove_seam(dummy_img, seam)
    # After removal, the image should have shape (3, 2, 3)
    assert new_img.shape == (3, 2, 3), "remove_seam shape incorrect"
    print("test_remove_seam passed")


if __name__ == "__main__":
    test_color_sum()
    test_energy()
    test_energy_matrix()
    test_find_path_and_retrieve_seam()
    test_remove_seam()
    print("All tests passed.")
