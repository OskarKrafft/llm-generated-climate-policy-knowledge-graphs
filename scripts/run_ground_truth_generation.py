import os
import sys
import json
from rdflib import Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD

# Add the project root to Python's path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)  # Make src importable

# Now import from src
from src.ground_truth_generation.generate_ground_truth import create_uri, normalize_actor_name, build_kg_for_article, parse_date_expression

# Define input and output directories
INPUT_DIR = "polianna-dataset/data/03a_processed_to_jsonl"
OUTPUT_DIR = "polianna-processed/turtle"

def main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """
    input_dir is the path that contains subfolders for each article.
    output_dir is the path where TTL files will be saved.
    This function iterates over each subfolder, builds the KG, saves a .ttl.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all directories (potential article folders) in the input directory
    article_dirs = []
    try:
        for item in os.listdir(input_dir):
            potential_article_dir = os.path.join(input_dir, item)
            
            # Check if it's a directory and contains the required files
            if (os.path.isdir(potential_article_dir) and 
                os.path.exists(os.path.join(potential_article_dir, "policy_info.json"))):
                article_dirs.append((item, potential_article_dir))
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found")
        return
    
    if not article_dirs:
        print(f"Warning: No valid article directories found in '{input_dir}'")
        return
    
    # Process each article directory
    for folder_name, article_path in article_dirs:
        try:
            # Build the graph
            g = build_kg_for_article(article_path)
            
            # Serialize to Turtle in the output directory
            ttl_out = os.path.join(output_dir, folder_name + ".ttl")
            g.serialize(destination=ttl_out, format="turtle")
            print(f"Saved {ttl_out}")
        except Exception as e:
            print(f"Error processing article '{folder_name}': {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments are provided, use them
        if len(sys.argv) < 3:
            print("Usage: python run_GTG.py <input_dir> <output_dir>")
            print(f"Using defaults: input={INPUT_DIR}, output={OUTPUT_DIR}")
        else:
            main(sys.argv[1], sys.argv[2])
    else:
        # Use default paths
        main()