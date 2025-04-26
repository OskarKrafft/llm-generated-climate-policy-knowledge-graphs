import os
import sys
from pathlib import Path
from rdflib import Graph, Namespace
from tqdm import tqdm
import argparse

# Add the project root to Python's path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Define namespaces (ensure this matches your ontology)
POLIANN = Namespace("https://polianna-kg.org/Ontology#")

def remove_fulltext(input_ttl_path, output_ttl_path):
    """Loads a TTL file, removes pol:fullText triples, and saves it."""
    try:
        g = Graph()
        g.parse(input_ttl_path, format="turtle")

        # Remove triples where the predicate is pol:fullText
        triples_to_remove = list(g.triples((None, POLIANN.fullText, None)))
        if triples_to_remove:
            for s, p, o in triples_to_remove:
                g.remove((s, p, o))
            
            # Save the modified graph
            g.serialize(destination=output_ttl_path, format="turtle")
            return True # Indicate that the file was modified
        else:
            # If no fullText triple was found, just copy the original file
            # Or you could choose to skip saving/copying if no change is needed
            g.serialize(destination=output_ttl_path, format="turtle")
            return False # Indicate no modification needed

    except Exception as e:
        print(f"  Error processing {input_ttl_path}: {e}")
        return None # Indicate error

def main(input_dir, output_dir):
    """
    Iterates through TTL files in input_dir, removes pol:fullText, 
    and saves them to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_path.resolve()}")
    print(f"Output directory: {output_path.resolve()}")
    
    ttl_files = list(input_path.glob("*.ttl"))
    
    if not ttl_files:
        print(f"No .ttl files found in {input_path}")
        return

    print(f"Found {len(ttl_files)} TTL files to process.")
    
    processed_count = 0
    modified_count = 0
    error_count = 0

    for ttl_file in tqdm(ttl_files, desc="Processing TTL files"):
        output_file_path = output_path / ttl_file.name
        result = remove_fulltext(ttl_file, output_file_path)
        
        if result is True:
            modified_count += 1
            processed_count += 1
        elif result is False:
            processed_count += 1 # Count as processed even if not modified
        else:
            error_count += 1
            
    print(f"\nProcessing complete.")
    print(f"  Total files processed: {processed_count}")
    print(f"  Files modified (fullText removed): {modified_count}")
    print(f"  Files with errors: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove pol:fullText triples from TTL files.")
    
    # Default paths relative to the project root
    default_input = project_root / "polianna-processed" / "turtle"
    default_output = project_root / "polianna-processed" / "turtle_no_fulltext"
    
    parser.add_argument("--input-dir", type=str, default=str(default_input),
                        help="Directory containing the original TTL files.")
    parser.add_argument("--output-dir", type=str, default=str(default_output),
                        help="Directory where TTL files without fullText will be saved.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)