import os
import csv

def list_mp4_with_base(parent_dir, base_dir):
    """
    This function recursively searches for `.mp4` files within a given directory and its subdirectories,
    and returns a list of their paths relative to a specified `parent_dir`, but with `base_dir` prepended 
    to each path.

    Args:
        parent_dir (str): The directory in which to start the search for `.mp4` files.
        base_dir (str): The base directory to prepend to each found `.mp4` file path.

    Returns:
        list: A list of full paths to `.mp4` files, with `base_dir` prepended to the relative paths.
    """
    mp4_paths = []  # Initialize an empty list to store file paths

    # Walk through the directory tree starting from parent_dir
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            # Check if the file ends with '.mp4' (case-insensitive)
            if file.lower().endswith('.mp4'):
                # Get the relative path from parent_dir to the file
                rel_path = os.path.relpath(os.path.join(root, file), parent_dir)

                # Prepend the base_dir to the relative path to get the full path
                full_path = os.path.normpath(os.path.join(base_dir, rel_path))
                mp4_paths.append(full_path)  # Add the full path to the list

    return mp4_paths  # Return the list of .mp4 file paths

if __name__ == "__main__":
    """
    This script will:
    1. Define paths for the parent directory and base directory.
    2. Use `list_mp4_with_base` to gather all `.mp4` file paths.
    3. Write these paths to a CSV file called `data.csv`.

    The CSV file will have a single column with the header `path`.
    """
    # ✅ Set your paths
    parent_directory = r"real_test_1k_sample"  # Directory to search for .mp4 files
    base_directory = r"./data/real_test_1k_sample/"  # Base directory to prepend to paths

    # ✅ Output CSV file name
    output_csv = "data.csv"  # CSV file to save the paths

    # ✅ Get list and write to CSV
    mp4_list = list_mp4_with_base(parent_directory, base_directory)  # Get list of .mp4 paths

    # Open the CSV file in write mode
    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)  # Create a CSV writer object
        writer.writerow(["path"])  # Write header row with column "path"
        
        # Write each path to the CSV, sorted alphabetically
        for path in sorted(mp4_list):
            writer.writerow([path])

    print(f"✅ {len(mp4_list)} mp4 paths written to {output_csv}")  # Print confirmation
