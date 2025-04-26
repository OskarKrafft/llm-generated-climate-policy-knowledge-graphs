import os
import sys
import argparse
from pathlib import Path
import shutil # Import shutil for copying files

# Add project root to sys.path to allow importing from src and ontology converter
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))
# Adjust the path to the ontology converter script relative to the project root
ontology_converter_path = project_root / "ontology/ontology-converter"
sys.path.append(str(ontology_converter_path))

try:
    # Attempt to import the conversion function from its location
    from scripts.ttl_to_jsonld import convert_ttl_to_jsonld, validate_jsonld
except ImportError as e:
    print(f"Error: Could not import 'convert_ttl_to_jsonld' or 'validate_jsonld'.")
    print(f"Attempted import from: {ontology_converter_path / 'scripts'}")
    print(f"Original error: {e}")
    print("Ensure 'ontology/ontology-converter/scripts/ttl_to_jsonld.py' exists and is importable.")
    sys.exit(1)

# --- Define Source and Target Base Directories ---
# Base directory where original training, test, validation data reside
DEFAULT_SOURCE_BASE_DIR = project_root
# Base directory for Experiment 2 JSON-LD data
DEFAULT_TARGET_BASE_DIR = project_root / "experiment-2-jsonld/data"
# --- End Define Directories ---

# Directories to process relative to the base directories
DATA_DIRS = ["training_data", "test_data", "validation_data"]

def process_data_directory(source_base_dir, target_base_dir, data_dir_name, validate=False):
    """
    Converts .ttl files and copies other relevant files (.txt, .json) from a source
    data directory to a target data directory.
    """
    source_dir = Path(source_base_dir) / data_dir_name
    target_dir = Path(target_base_dir) / data_dir_name

    print(f"\n--- Processing Directory: {data_dir_name} ---")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    if not source_dir.is_dir():
        print(f"Warning: Source directory '{source_dir}' not found. Skipping.")
        return 0, 0, 0 # converted, copied, errors

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured target directory exists: {target_dir}")

    converted_count = 0
    copied_count = 0
    error_count = 0

    for item in source_dir.iterdir():
        source_item_path = item
        target_item_path = target_dir / item.name

        if item.is_file():
            if item.suffix == ".ttl":
                # Convert .ttl to .jsonld
                jsonld_filename = f"{item.stem}.jsonld"
                output_file = target_dir / jsonld_filename
                input_file = str(source_item_path)

                print(f"  Converting '{item.name}' -> '{jsonld_filename}'...")
                try:
                    jsonld_data = convert_ttl_to_jsonld(input_file, str(output_file))
                    if validate:
                        # Temporarily suppress print from validate_jsonld for cleaner batch output
                        original_stdout = sys.stdout
                        sys.stdout = open(os.devnull, 'w')
                        is_valid = validate_jsonld(jsonld_data)
                        sys.stdout.close()
                        sys.stdout = original_stdout # Restore stdout
                        if not is_valid:
                            print(f"    Validation failed for {jsonld_filename}")
                            # Keep the file but note the validation failure
                    converted_count += 1
                    print(f"    Successfully converted and saved to '{output_file}'")
                except Exception as e:
                    print(f"    Error converting '{item.name}': {e}")
                    error_count += 1
                    # Optionally remove the potentially incomplete output file
                    if output_file.exists():
                        output_file.unlink()
            elif item.suffix in [".txt", ".json"]: # Copy .txt and .json files
                 print(f"  Copying '{item.name}' -> '{target_item_path}'...")
                 try:
                     shutil.copy2(source_item_path, target_item_path) # copy2 preserves metadata
                     copied_count += 1
                     print(f"    Successfully copied.")
                 except Exception as e:
                     print(f"    Error copying '{item.name}': {e}")
                     error_count += 1
            else:
                print(f"  Skipping non-TTL/TXT/JSON file: {item.name}")
        elif item.is_dir():
            # Process subdirectories recursively instead of skipping
            print(f"  Processing subdirectory: {item.name}")
            # Create corresponding target subdirectory
            target_subdir = target_dir / item.name
            target_subdir.mkdir(parents=True, exist_ok=True)
            
            # Process all files in the subdirectory
            for subitem in source_item_path.iterdir():
                if subitem.is_file():
                    source_subitem_path = subitem
                    target_subitem_path = target_subdir / subitem.name
                    
                    if subitem.suffix == ".ttl":
                        # Convert .ttl to .jsonld
                        jsonld_filename = f"{subitem.stem}.jsonld"
                        output_file = target_subdir / jsonld_filename
                        input_file = str(source_subitem_path)
                        
                        print(f"    Converting '{subitem.name}' -> '{jsonld_filename}'...")
                        try:
                            jsonld_data = convert_ttl_to_jsonld(input_file, str(output_file))
                            if validate:
                                # Validation code...
                                original_stdout = sys.stdout
                                sys.stdout = open(os.devnull, 'w')
                                is_valid = validate_jsonld(jsonld_data)
                                sys.stdout.close()
                                sys.stdout = original_stdout
                                if not is_valid:
                                    print(f"      Validation failed for {jsonld_filename}")
                            converted_count += 1
                            print(f"      Successfully converted and saved to '{output_file}'")
                        except Exception as e:
                            print(f"      Error converting '{subitem.name}': {e}")
                            error_count += 1
                            if output_file.exists():
                                output_file.unlink()
                    elif subitem.suffix in [".txt", ".json"]:
                        print(f"    Copying '{subitem.name}' -> '{target_subitem_path}'...")
                        try:
                            shutil.copy2(source_subitem_path, target_subitem_path)
                            copied_count += 1
                            print(f"      Successfully copied.")
                        except Exception as e:
                            print(f"      Error copying '{subitem.name}': {e}")
                            error_count += 1
                    else:
                        print(f"    Skipping non-TTL/TXT/JSON file: {subitem.name}")

    print(f"--- Finished Processing: {data_dir_name} ---")
    print(f"  Converted TTL files: {converted_count}")
    print(f"  Copied other files (.txt, .json): {copied_count}") # Updated summary message
    print(f"  Errors encountered: {error_count}")
    return converted_count, copied_count, error_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch convert Turtle (.ttl) files to JSON-LD (.jsonld) and copy relevant "
                    "files (.txt, .json) for training, test, and validation sets into the experiment directory."
    )
    parser.add_argument(
        "--source-base-dir",
        type=str,
        default=str(DEFAULT_SOURCE_BASE_DIR),
        help=f"Base directory containing the source data folders (training_data, etc.) (default: {DEFAULT_SOURCE_BASE_DIR})"
    )
    parser.add_argument(
        "--target-base-dir",
        type=str,
        default=str(DEFAULT_TARGET_BASE_DIR),
        help=f"Base directory where the output JSON-LD data folders will be created (default: {DEFAULT_TARGET_BASE_DIR})"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated JSON-LD files after conversion."
    )

    args = parser.parse_args()

    total_converted = 0
    total_copied = 0
    total_errors = 0

    print("Starting data preparation for Experiment 2 (JSON-LD)...")

    for data_dir in DATA_DIRS:
        converted, copied, errors = process_data_directory(
            args.source_base_dir,
            args.target_base_dir,
            data_dir,
            args.validate
        )
        total_converted += converted
        total_copied += copied
        total_errors += errors

    print("\n======================================")
    print("Overall Data Preparation Summary:")
    print(f"  Total TTL files converted: {total_converted}")
    print(f"  Total other files copied (.txt, .json): {total_copied}") # Updated summary message
    print(f"  Total errors encountered: {total_errors}")
    print(f"Target data location: {Path(args.target_base_dir).resolve()}")
    print("======================================")