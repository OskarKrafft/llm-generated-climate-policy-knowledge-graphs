import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from rdflib import Graph, Literal, URIRef # Added URIRef

# Add the project root to Python's path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.append(str(project_root))

# --- Define Namespaces (copied from generate_raw_triplets.py) ---
namespaces = {
    '': 'https://polianna-kg.org/Ontology#',
    'eli': 'http://data.europa.eu/eli/ontology#',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'eurovoc': 'http://eurovoc.europa.eu/'
}
# --- End Namespaces ---

# --- extract_triplets function (copied from generate_raw_triplets.py) ---
def extract_triplets(ttl_path):
    g = Graph()
    g.parse(ttl_path, format='turtle') # Expects string path
    triplets = []
    base_article_id = os.path.basename(ttl_path).replace('.ttl', '') # Uses os.path
    base_article_uri_part = namespaces[''] + base_article_id

    for s, p, o in g:
        def simplify_uri(uri_str):
            # Rule 0: Special case for rdf:type predicate
            if uri_str == namespaces['rdf'] + 'type':
                return "type"

            # Rule 1: Exact match for base article URI
            if uri_str == base_article_uri_part:
                return "Article"
            # Rule 2: Starts with base article URI + '_' (suffix)
            if uri_str.startswith(base_article_uri_part + '_'):
                return uri_str.split(base_article_uri_part + '_', 1)[1]

            # Rule 3: Check non-base namespaces first
            for prefix, ns in namespaces.items():
                if prefix and uri_str.startswith(ns): # Only check non-empty prefixes
                    local = uri_str[len(ns):]
                    return f"{prefix}:{local}" if local else prefix

            # Rule 4: Check base namespace ('')
            base_ns = namespaces.get('', None)
            if base_ns and uri_str.startswith(base_ns):
                local = uri_str[len(base_ns):]
                return local

            # Fallback rules (less specific)
            if '#' in uri_str:
                if uri_str == base_ns:
                     return ":"
                return uri_str.split('#')[-1]
            if '/' in uri_str:
                 for prefix, ns in namespaces.items():
                     if uri_str == ns:
                         return prefix
                 return uri_str.rsplit('/', 1)[-1]

            return uri_str # Return original if no rule applied

        subj = simplify_uri(str(s))
        pred = simplify_uri(str(p))

        if isinstance(o, Literal):
            obj = str(o)
        else:
            obj = simplify_uri(str(o))
        triplets.append({"s": subj, "p": pred, "o": obj})
    return triplets
# --- End extract_triplets function ---


# --- Define Directories ---
DEFAULT_SOURCE_BASE_DIR = project_root
DEFAULT_TARGET_BASE_DIR = project_root / "experiment-2-raw/data"
DATA_DIRS = ["training_data", "test_data", "validation_data"]
# --- End Define Directories ---

def process_data_directory(source_base_dir, target_base_dir, data_dir_name):
    """
    Converts .ttl files to raw triplet .json files and copies other relevant
    files (.txt, .json) from a source data directory to a target data directory.
    Prioritizes *_no_fulltext.ttl for test/validation sets.
    """
    source_dir = Path(source_base_dir) / data_dir_name
    target_dir = Path(target_base_dir) / data_dir_name

    print(f"\n--- Processing Directory: {data_dir_name} ---")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    if not source_dir.is_dir():
        print(f"Warning: Source directory '{source_dir}' not found. Skipping.")
        return 0, 0, 0 # converted, copied, errors

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured target directory exists: {target_dir}")

    converted_count = 0
    copied_count = 0
    error_count = 0

    for article_item in source_dir.iterdir():
        if not article_item.is_dir():
            continue

        article_id = article_item.name
        target_article_dir = target_dir / article_id
        target_article_dir.mkdir(parents=True, exist_ok=True)

        # --- Determine the correct source TTL file and target JSON name ---
        ttl_file_path = None
        # Target JSON name is always based on the article_id
        output_json_name = f"{article_id}.json"
        output_json_path = target_article_dir / output_json_name

        if data_dir_name in ["test_data"]:
            # Prioritize the _no_fulltext.ttl file for test
            potential_ttl_path = article_item / f"{article_id}_no_fulltext.ttl"
            if potential_ttl_path.is_file():
                ttl_file_path = potential_ttl_path
            else:
                # Fallback or error if _no_fulltext.ttl is missing
                fallback_ttl_path = article_item / f"{article_id}.ttl"
                if fallback_ttl_path.is_file():
                    print(f"  Warning: Expected '{article_id}_no_fulltext.ttl' not found in {article_item}. Falling back to '{article_id}.ttl'.")
                    ttl_file_path = fallback_ttl_path
                else:
                     print(f"  Error: No suitable TTL file found for {article_id} in {article_item} for {data_dir_name}.")
                     error_count += 1
                     continue # Skip this article if no TTL found
        else: # For training_data or validation_data or other directories
            # Use the standard article_id.ttl file
            potential_ttl_path = article_item / f"{article_id}.ttl"
            if potential_ttl_path.is_file():
                ttl_file_path = potential_ttl_path
            else:
                # Fallback for training if needed (optional)
                fallback_ttl_path = article_item / f"{article_id}_no_fulltext.ttl"
                if fallback_ttl_path.is_file():
                     print(f"  Warning: Standard '{article_id}.ttl' not found in {article_item}. Falling back to '{article_id}_no_fulltext.ttl'.")
                     ttl_file_path = fallback_ttl_path
                else:
                    print(f"  Error: No suitable TTL file found for {article_id} in {article_item} for {data_dir_name}.")
                    error_count += 1
                    continue # Skip this article if no TTL found

        # --- Convert the selected TTL file ---
        if ttl_file_path:
            try:
                # Pass path as string
                triplets = extract_triplets(str(ttl_file_path))
                with open(output_json_path, 'w') as f:
                    json.dump(triplets, f, indent=2)
                converted_count += 1
            except Exception as e:
                print(f"    ERROR converting {ttl_file_path.name}: {e}")
                error_count += 1

        # --- Copy other relevant files (.txt, .json) ---
        for file_path in article_item.iterdir():
            # Skip the source TTL file itself and the target JSON file name
            if file_path.suffix == ".ttl" or file_path.name == output_json_name:
                 continue

            if file_path.suffix in [".txt", ".json"]:
                target_file_path = target_article_dir / file_path.name
                try:
                    shutil.copy2(file_path, target_file_path)
                    copied_count += 1
                except Exception as e:
                    print(f"    ERROR copying {file_path.name}: {e}")
                    error_count += 1

    print(f"--- Finished {data_dir_name}: Converted={converted_count}, Copied={copied_count}, Errors={error_count} ---")
    return converted_count, copied_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert ground truth Turtle (.ttl) files to raw triplet JSON (.json) "
                    "and copy relevant files (.txt, .json) for training, test, and validation sets "
                    "into the experiment-2-raw directory."
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
        help=f"Base directory where the output raw triplet data folders will be created (default: {DEFAULT_TARGET_BASE_DIR})"
    )

    args = parser.parse_args()

    source_base = Path(args.source_base_dir)
    target_base = Path(args.target_base_dir)

    total_converted = 0
    total_copied = 0
    total_errors = 0

    print("Starting data preparation for Experiment 2 (Raw Triplets)...")

    for data_dir in DATA_DIRS:
        converted, copied, errors = process_data_directory(
            source_base,
            target_base,
            data_dir
        )
        total_converted += converted
        total_copied += copied
        total_errors += errors

    print("\n======================================")
    print("Overall Data Preparation Summary:")
    print(f"  Total TTL files converted to JSON triplets: {total_converted}")
    print(f"  Total other files copied (.txt, .json): {total_copied}")
    print(f"  Total errors encountered: {total_errors}")
    print(f"Output data located in: {target_base}")
    print("======================================")

if __name__ == "__main__":
    main()