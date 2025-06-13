import os
import subprocess
from tqdm.notebook import tqdm
import IPython.display as display

def get_subfolders(parent_folder):
    """
    Get a list of all subfolders in the given parent folder.
    Handles Windows path separators correctly.
    
    Args:
        parent_folder: Path to the parent folder
        
    Returns:
        List of absolute paths to all subfolders
    """
    # Ensure parent_folder is an absolute path with proper separators
    parent_folder = os.path.abspath(os.path.normpath(parent_folder))
    
    # Get all items in the parent folder
    subfolders = []
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        # Check if it's a directory
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    
    return sorted(subfolders)  # Sort for consistent order

# Input: Parent folder containing all the directories to process
parent_folder = "path/to/parent_folder"  # UPDATE THIS PATH to your input directory

# Get all subfolders
directories = get_subfolders(parent_folder)
print(f"Found {len(directories)} directories to process:")
for directory in directories:
    print(f" - {os.path.basename(directory)}")

# Output directory (optional)
output_dir = "path/to/output"  # UPDATE THIS PATH to your output directory or set to None

# Additional parameters
od_model = "yolov12"
use_simplify = True

# Display a confirmation before running
confirmation = input(f"Process {len(directories)} directories? (y/n): ")

if confirmation.lower() == 'y':
    # Run inference on each directory
    for directory in tqdm(directories, desc="Processing directories"):
        cmd = [
            "python", "-m", "lnp_mod.core.inference",
            "-i", directory,
            "--od_model", od_model
        ]
        
        # Add output dir if specified
        if output_dir:
            # Create directory-specific output folder
            dir_output = os.path.join(output_dir, os.path.basename(directory))
            cmd.extend(["-o", dir_output])
        
        # Add other parameters
        if use_simplify:
            cmd.append("--simplify")
        
        # Display current directory being processed
        display.clear_output(wait=True)
        print(f"Processing directory: {directory}")
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Display command output in the notebook
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Error processing {directory}:")
            print(result.stderr)
    
    display.clear_output(wait=True)
    print("All directories processed!")
else:
    print("Operation cancelled.")