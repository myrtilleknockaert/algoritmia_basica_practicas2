#!/bin/bash
# This script automates the execution of the seam carving algorithm.
# Usage: ./ejecutar.sh <number_of_seams> <image_file> <output_directory>
# Example: ./ejecutar.sh 5 photo.png /tmp/

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <number_of_seams> <image_file> <output_directory>"
    exit 1
fi

NUM_SEAMS="$1"
IMAGE_FILE="$2"
OUTPUT_DIR="$3"

# Verify that the image file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo "Error: Image file '$IMAGE_FILE' not found."
    exit 1
fi

# Create output directory if it does not exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory '$OUTPUT_DIR' does not exist. Creating it..."
    mkdir -p "$OUTPUT_DIR"
fi

# Ensure the output directory path ends with a slash
case "$OUTPUT_DIR" in
    */) ;;  # Already ends with /
    *) OUTPUT_DIR="${OUTPUT_DIR}/" ;;
esac

# Execute the Python seam carving script
python3 seam_carving.py "$NUM_SEAMS" "$IMAGE_FILE" "$OUTPUT_DIR"
