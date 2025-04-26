#!/usr/bin/env python3
import sys
import json
import argparse
import os
from pathlib import Path
from rdflib import Graph, Literal, URIRef

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

            # Fallback rules
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

def main():
    parser = argparse.ArgumentParser(
        description="Convert one or more Turtle (.ttl) files to raw triplet JSON (.json) files in the same directory."
    )
    parser.add_argument(
        "ttl_files",
        metavar="TTL_FILE",
        type=str,
        nargs='+',
        help="Path(s) to the Turtle file(s) to convert."
    )

    args = parser.parse_args()

    for ttl_file_str in args.ttl_files:
        ttl_path = Path(ttl_file_str).resolve() # Use resolve for absolute path

        if not ttl_path.is_file():
            print(f"Error: File not found: {ttl_path}", file=sys.stderr)
            continue
        if ttl_path.suffix.lower() != ".ttl":
            print(f"Error: File is not a .ttl file: {ttl_path}", file=sys.stderr)
            continue

        # Determine output path (same directory, .json extension)
        output_json_path = ttl_path.with_suffix('.json')

        print(f"Processing: {ttl_path.name} -> {output_json_path.name}")

        try:
            # Pass path as string
            triplets = extract_triplets(str(ttl_path))
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(triplets, f, indent=2, ensure_ascii=False)
            print(f"  Successfully generated: {output_json_path}")
        except Exception as e:
            print(f"  Error processing {ttl_path.name}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()