import os
from pathlib import Path
import rdflib
import json
import argparse # Keep argparse for flexibility

# --- Configuration ---
DEFAULT_TEST_DATA_DIR = Path("experiment-2-jsonld/data/test_data")

def test_parse_single_jsonld_file(file_path):
    """Attempts to parse a single JSON-LD file."""
    print(f"\n--- Attempting to parse: {file_path} ---")
    if not file_path.is_file():
        print(f"ERROR: File not found: {file_path}")
        return False

    g = rdflib.Graph()
    try:
        # Attempt to parse the file as JSON-LD
        g.parse(str(file_path), format="json-ld")
        print(f"SUCCESS: Parsed {file_path.name} ({len(g)} triples)")
        return True
    except json.JSONDecodeError as json_err:
        print(f"ERROR: Invalid JSON in {file_path.name}")
        print(f"   Details: {json_err}")
        return False
    except Exception as e:
        # Catch other potential rdflib parsing errors
        print(f"ERROR: Failed to parse {file_path.name} as JSON-LD")
        print(f"   Details: {type(e).__name__}: {e}")
        return False

def test_parse_jsonld_recursive(directory):
    """
    Recursively finds and attempts to parse all .jsonld files
    within the given directory and its subdirectories.
    """
    print(f"Recursively checking JSON-LD files in: {directory.resolve()}")
    if not directory.is_dir():
        print(f"Error: Directory not found: {directory}")
        return

    found_files = 0
    parsed_files = 0
    error_files = 0

    # Use rglob to find all .jsonld files recursively
    for item_path in directory.rglob('*.jsonld'):
        if item_path.is_file(): # Ensure it's actually a file
            found_files += 1
            if test_parse_single_jsonld_file(item_path):
                parsed_files += 1
            else:
                error_files += 1

    print("\n--- Recursive Summary ---")
    print(f"Found .jsonld files: {found_files}")
    print(f"Successfully parsed: {parsed_files}")
    print(f"Failed to parse:    {error_files}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test parsing of JSON-LD files.")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a specific .jsonld file to test."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_TEST_DATA_DIR),
        help=f"Directory containing .jsonld files to test recursively (default: {DEFAULT_TEST_DATA_DIR})."
    )
    args = parser.parse_args()

    if args.file:
        # Test a single file if provided
        file_to_test = Path(args.file)
        test_parse_single_jsonld_file(file_to_test)
    else:
        # Otherwise, test the directory recursively
        directory_to_test = Path(args.dir)
        test_parse_jsonld_recursive(directory_to_test)