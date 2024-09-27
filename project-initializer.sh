#!/bin/bash

# Exit script on any error
set -e

# Prompt the user for the project name
read -p "Enter the project name: " PROJECT_NAME

# Use Python 3 by default, adjust if needed
PYTHON_VERSION="python" 

# Create the project directory
echo "Creating project directory: $PROJECT_NAME"
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Set up a virtual environment
echo "Setting up virtual environment..."
$PYTHON_VERSION -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/Scripts/activate

# Upgrade pip to the latest version
# echo "Upgrading pip..."
# pip install --upgrade pip

# Install necessary packages
echo "Installing required packages..."
pip install tensorflow scikit-learn numpy matplotlib glob2

echo "Creating a requirements.txt file..."
pip freeze > requirements.txt

# Notify user of completion
echo "Project setup complete!"
echo "To activate the virtual environment in the future, run: source .venv/bin/activate"

touch app.py
