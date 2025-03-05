#!/bin/bash
# ejecutar.sh
# This script prepares the environment for running the seam carving command.
# It will check for Python3 and pip, create a virtual environment,
# install necessary dependencies, and set the proper permissions.

# Check if Python3 is installed
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Check if pip3 is installed
if ! command -v pip3 >/dev/null 2>&1; then
    echo "Error: pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install the required Python packages
echo "Installing required packages: numpy, matplotlib..."
pip install numpy matplotlib

# Ensure the main script is executable (assuming it's named seam_carving.py)
chmod +x seam_carving.py

echo "Environment is ready!"
echo "You can now run the command:"
echo "  ./seam_carving.py <num_seams> <image_file> <output_directory>"
echo "For example:"
echo "  ./seam_carving.py 50 photo.png ./"

# End of script
