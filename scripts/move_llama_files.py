import os
import shutil
import argparse
import sys

def move_matching_files(source_dir, target_dir, name_part1, name_part2):
    """
    Searches subfolders of source_dir for files containing name_part1 AND name_part2
    in their name and moves them to target_dir, preserving subfolder structure.
    """
    print(f"Source Directory: {os.path.abspath(source_dir)}")
    print(f"Target Directory: {os.path.abspath(target_dir)}")
    print(f"Searching for files containing BOTH '{name_part1}' AND '{name_part2}'...")
    print("-" * 40)

    moved_count = 0

    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # Iterate through the source directory tree
    for root, _, files in os.walk(source_dir):
        for filename in files:
            # Check if both parts are in the filename
            if name_part1 in filename and name_part2 in filename:
                try:
                    # Construct full source path
                    src_file_path = os.path.join(root, filename)

                    # Determine path relative to the source directory
                    relative_path = os.path.relpath(src_file_path, source_dir)

                    # Construct full destination path preserving structure
                    dest_file_path = os.path.join(target_dir, relative_path)

                    # Ensure destination subdirectory exists
                    dest_subdir = os.path.dirname(dest_file_path)
                    if not os.path.exists(dest_subdir):
                        os.makedirs(dest_subdir, exist_ok=True)
                        print(f"Created directory: {dest_subdir}")

                    # Move the file
                    print(f"Moving: '{relative_path}'")
                    shutil.move(src_file_path, dest_file_path)
                    moved_count += 1

                except Exception as e:
                    print(f"  Error moving '{filename}': {e}", file=sys.stderr)

    print("-" * 40)
    print(f"Operation complete. Moved {moved_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move files from source to target if filename contains specific strings."
    )
    parser.add_argument(
        "source_directory",
        help="The directory to search within (including subfolders)."
    )
    parser.add_argument(
        "target_directory",
        help="The directory where matching files will be moved."
    )

    args = parser.parse_args()

    # --- Define the specific strings to search for in filenames ---
    # NOTE: Using ':' in filenames can sometimes cause issues on certain systems,
    # but assuming it works based on your example.
    search_string1 = "qwen2.5-coder"
    search_string2 = "20250421" # As specified

    move_matching_files(
        args.source_directory,
        args.target_directory,
        search_string1,
        search_string2
    )