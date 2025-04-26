#!/usr/bin/env python
"""
Knowledge Graph Evaluation Script

This script runs comprehensive evaluations on generated knowledge graphs, comparing them to 
ground truth data and producing detailed analysis by model and strategy.

Example usage:
    # Basic evaluation
    python run_evaluation.py
    
    # Evaluate with specific paths and filtering options
    python run_evaluation.py --results-dir ../experiment-1/results --ground-truth-dir ../test_data
    
    # Collect problematic files for analysis
    python run_evaluation.py --collect-problematic --f1-threshold 0.2
    
    # Generate detailed report (without visualizations)
    python run_evaluation.py --report-dir ../experiment-1/reports/evaluation
"""

import os
import sys
import glob
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import datetime
import json
from rdflib import Graph
import re  # Add import for regex

# Add the project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.evaluation.evaluation import (
    KnowledgeGraphEvaluator,
    collect_problematic_files,
    generate_evaluation_report
)


def find_ground_truth_file(root_dir, article_id, extension=".ttl"):
    """Find the ground truth file for an article, using the specified extension."""
    article_dir = os.path.join(root_dir, article_id)
    if not os.path.exists(article_dir):
        return None
    # Try direct _no_fulltext file
    gt_path = os.path.join(article_dir, f"{article_id}_no_fulltext{extension}")
    if os.path.exists(gt_path):
        return gt_path
    # Fallback to any file with given extension
    files = glob.glob(os.path.join(article_dir, f"*{extension}"))
    return files[0] if files else None


def extract_metadata_from_filename(filename):
    """
    Extract metadata (article ID, model, strategy) from a filename.
    
    Expected patterns: 
    - {celex}_{model}_{strategy}_{timestamp}.ttl
    - {celex}_{model}-{date}_{strategy}_{timestamp}.ttl
    """
    parts = filename.split('_')
    
    metadata = {
        'model': 'unknown',
        'strategy': 'unknown',
        'timestamp': None
    }
    
    # Try to identify the strategy
    for part in parts:
        if part in ["zero-shot", "one-shot", "few-shot"]:
            metadata['strategy'] = part
            break
    
    # Try to identify the model (usually after the article ID, which is the first part)
    if len(parts) >= 3:
        # Check if the second part contains model information
        model_part = parts[1]
        
        # Handle modern model formats with dates like o3-mini-2025-01-31
        if "-" in model_part:
            # This could be a model name with a date or version
            metadata['model'] = model_part
            
        # Handle traditional model names
        elif model_part in ['gpt3', 'gpt4', 'gpt-4', 'llama', 'claude', 'o3']:
            metadata['model'] = model_part
            
            # Check if the next part is part of the model name (like gpt-4-0613)
            if len(parts) >= 4 and parts[2] not in ["zero-shot", "one-shot", "few-shot"]:
                # Could be a version number or additional model identifier
                if not parts[2].startswith('202'):  # Not a timestamp
                    metadata['model'] = f"{model_part}-{parts[2]}"
                    
        # Handle model names with colons (like llama3.1:70b)
        elif ':' in model_part:
            metadata['model'] = model_part
    
    # Special case for GPT-4o with dates (gpt-4o-2024-08-06)
    if metadata['model'] == 'unknown' and len(parts) >= 3:
        # Try to reconstruct from multiple parts
        potential_model = parts[1]
        if potential_model in ['gpt', 'gpt4', 'gpt-4', 'o3']:
            # Check if next part completes the model name
            if len(parts) >= 4:
                if parts[2] in ['4o', '4', '3.5', 'o']:
                    potential_model = f"{potential_model}-{parts[2]}"
                    # Check if there's a date part
                    if len(parts) >= 5 and parts[3].startswith('202'):  # Year format
                        potential_model = f"{potential_model}-{parts[3]}"
                    metadata['model'] = potential_model
    
    return metadata


def load_ontology():
    """Load the POLIANNA ontology if available."""
    ontology_path = os.path.join(project_root, "ontology", "ontology_v17_no_fulltext.ttl")
    
    if os.path.exists(ontology_path):
        try:
            ontology = Graph()
            ontology.parse(ontology_path, format="turtle")
            print(f"Loaded ontology with {len(ontology)} triples")
            return ontology
        except Exception as e:
            print(f"Error loading ontology: {e}")
    else:
        print(f"Ontology file not found: {ontology_path}")
    
    return None


def main(args):
    """Run the evaluation process."""
    # Create timestamp for output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine file extensions based on input format
    generated_extension = ".ttl" if args.generated_format == 'turtle' else ".jsonld"
    ground_truth_extension = ".ttl" if args.ground_truth_format == 'turtle' else ".jsonld"

    # Initialize evaluator
    evaluator = KnowledgeGraphEvaluator()
    
    # Load ontology if available
    ontology = load_ontology()
    if ontology:
        evaluator.ontology = ontology
    
    # Find all article directories in results folder
    article_dirs = []
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        if os.path.isdir(item_path) and item.startswith(args.article_prefix):
            article_dirs.append(item)
    
    total_articles = len(article_dirs)  # Get total number of articles
    print(f"Found {total_articles} article result directories")
    
    # Skip if no articles found
    if not article_dirs:
        print("No article directories found. Exiting.")
        return
    
    # Results storage
    results = []
    
    # Set up progress bar
    progress_bar = tqdm(article_dirs, desc="Evaluating articles")
    
    # Process each article
    for article_id in progress_bar:
        # Update progress description
        progress_bar.set_description(f"Evaluating {article_id}")
        
        # Get article results directory
        article_results_dir = os.path.join(args.results_dir, article_id)
        
        # Find and load ground truth using specified format
        ground_truth_path = find_ground_truth_file(args.ground_truth_dir, article_id, extension=ground_truth_extension)
        if not ground_truth_path:
            print(f"  Skipping {article_id}: Ground truth ({ground_truth_extension}) not found")
            continue
        ground_truth_graph = Graph()
        try:
            ground_truth_graph.parse(ground_truth_path, format=args.ground_truth_format)
        except Exception as e:
            print(f"  Skipping {article_id}: Error parsing ground truth {ground_truth_path} as {args.ground_truth_format}: {e}")
            continue
        
        # Find all generated files with the specified extension
        result_files = []
        if os.path.isdir(article_results_dir):
            for file in os.listdir(article_results_dir):
                if file.endswith(generated_extension) and os.path.isfile(os.path.join(article_results_dir, file)):
                    if args.model_filter and args.model_filter not in file: continue
                    if args.strategy_filter and args.strategy_filter not in file: continue
                    result_files.append(os.path.join(article_results_dir, file))
        
        # Evaluate each generated file
        for result_file in result_files:
            file_name = os.path.basename(result_file)
            metadata = extract_metadata_from_filename(file_name)
            cleaned = False
            fixed_jsonld = False  # Flag to track if JSON-LD was fixed
            try:
                file_content = None
                if args.generated_format == 'turtle':
                    cleaned = evaluator.clean_ttl_file(result_file)
                elif args.generated_format == 'json-ld':
                    # First clean the file using the evaluator's method
                    cleaned = evaluator.clean_jsonld_file(result_file)
                    
                    # After cleaning, read for additional fixes if needed
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        # Additional fixes if still needed
                        if '"@voca: ' in file_content:
                            fixed_content = file_content.replace('"@voca: ', '"@vocab": "')
                            if fixed_content != file_content:
                                fixed_jsonld = True
                                file_content = fixed_content
                    except Exception as read_err:
                        raise ValueError(f"Error reading JSON-LD file {result_file}: {read_err}") from read_err

                g = Graph()
                # Parse from file content if read and potentially fixed, otherwise parse from path
                if file_content is not None and args.generated_format == 'json-ld':
                    g.parse(data=file_content, format=args.generated_format)
                else:
                    g.parse(result_file, format=args.generated_format)
                    
                metrics = evaluator.evaluate_comprehensive(g, ground_truth_graph)
                metrics["triple_count"] = len(g)
                metrics["file_size"] = os.path.getsize(result_file)
                metrics["cleaned"] = cleaned  # Keep track of turtle cleaning
                metrics["fixed_jsonld"] = fixed_jsonld  # Add flag for jsonld fixing
            except Exception as e:
                metrics = {
                    "is_valid": False,
                    "syntax_errors": 1,
                    "syntax_error_message": str(e),
                    "file_path": result_file,
                    "file_size": os.path.getsize(result_file) if os.path.exists(result_file) else 0,
                    "triple_count": 0,
                    "cleaned": cleaned,
                    "fixed_jsonld": fixed_jsonld  # Also track if fixing was attempted before error
                }
            # Add common metadata
            metrics.update({
                "article_id": article_id,
                "model": metadata.get("model"),
                "strategy": metadata.get("strategy"),
                "file_name": file_name
            })
            results.append(metrics)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print(f"Completed evaluation of {len(results)} files found.")

    # --- Start Modification: Create a complete DataFrame ---
    if not results_df.empty and total_articles > 0:
        unique_models = results_df['model'].unique()
        unique_strategies = results_df['strategy'].unique()

        # Create all expected combinations
        expected_index = pd.MultiIndex.from_product(
            [article_dirs, unique_models, unique_strategies], 
            names=['article_id', 'model', 'strategy']
        )
        expected_df = pd.DataFrame(index=expected_index).reset_index()

        # Merge actual results onto the expected dataframe
        # Keep only columns that exist in results_df for merging, plus the key columns
        merge_cols = ['article_id', 'model', 'strategy'] + [col for col in results_df.columns if col not in ['article_id', 'model', 'strategy']]
        results_df = pd.merge(
            expected_df, 
            results_df[merge_cols], # Select columns explicitly
            on=['article_id', 'model', 'strategy'], 
            how='left'
        )

        # Fill NaNs for missing files with appropriate defaults
        fill_values = {
            'is_valid': False,
            'syntax_errors': 0,
            'triple_count': 0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'file_size': 0,
            'cleaned': False,
            'fixed_jsonld': False,
            'syntax_error_message': 'File not generated or found',
            'ground_truth_triple_count': 0, # Add default for GT count
            # Add defaults for category metrics (assuming 0/0.0 is appropriate)
            'tp_core': 0, 'fp_core': 0, 'fn_core': 0, 'precision_core': 0.0, 'recall_core': 0.0, 'f1_score_core': 0.0,
            'tp_policy_char': 0, 'fp_policy_char': 0, 'fn_policy_char': 0, 'precision_policy_char': 0.0, 'recall_policy_char': 0.0, 'f1_score_policy_char': 0.0,
            'tp_compliance_char': 0, 'fp_compliance_char': 0, 'fn_compliance_char': 0, 'precision_compliance_char': 0.0, 'recall_compliance_char': 0.0, 'f1_score_compliance_char': 0.0,
            'tp_actor': 0, 'fp_actor': 0, 'fn_actor': 0, 'precision_actor': 0.0, 'recall_actor': 0.0, 'f1_score_actor': 0.0,
            'tp_time': 0, 'fp_time': 0, 'fn_time': 0, 'precision_time': 0.0, 'recall_time': 0.0, 'f1_score_time': 0.0,
            'tp_other': 0, 'fp_other': 0, 'fn_other': 0, 'precision_other': 0.0, 'recall_other': 0.0, 'f1_score_other': 0.0,
            # Add defaults for other metrics if they might be missing
            'property_precision': 0.0, 'property_recall': 0.0, 'property_f1_score': 0.0, 'property_true_positives': 0, 'property_false_positives': 0, 'property_false_negatives': 0, 'property_diversity': 0,
            'class_precision': 0.0, 'class_recall': 0.0, 'class_f1_score': 0.0, 'class_true_positives': 0, 'class_false_positives': 0, 'class_false_negatives': 0,
            'entity_precision': 0.0, 'entity_recall': 0.0, 'entity_f1_score': 0.0, 'entity_true_positives': 0, 'entity_false_positives': 0, 'entity_false_negatives': 0,
            'ontology_consistency_score': 0.0, 'constraint_violations': 0, 'ontology_conforms': False,
            # Add defaults for relationship metrics (example for one, repeat for others)
            'rel_precision_contains_instrument': 0.0, 'rel_recall_contains_instrument': 0.0, 'rel_f1_score_contains_instrument': 0.0, 'rel_true_positives_contains_instrument': 0, 'rel_false_positives_contains_instrument': 0, 'rel_false_negatives_contains_instrument': 0,
            'rel_precision_contains_objective': 0.0, 'rel_recall_contains_objective': 0.0, 'rel_f1_score_contains_objective': 0.0, 'rel_true_positives_contains_objective': 0, 'rel_false_positives_contains_objective': 0, 'rel_false_negatives_contains_objective': 0,
            'rel_precision_contains_monitoring_form': 0.0, 'rel_recall_contains_monitoring_form': 0.0, 'rel_f1_score_contains_monitoring_form': 0.0, 'rel_true_positives_contains_monitoring_form': 0, 'rel_false_positives_contains_monitoring_form': 0, 'rel_false_negatives_contains_monitoring_form': 0,
            'rel_precision_contains_sanctioning_form': 0.0, 'rel_recall_contains_sanctioning_form': 0.0, 'rel_f1_score_contains_sanctioning_form': 0.0, 'rel_true_positives_contains_sanctioning_form': 0, 'rel_false_positives_contains_sanctioning_form': 0, 'rel_false_negatives_contains_sanctioning_form': 0,
            'rel_precision_addresses': 0.0, 'rel_recall_addresses': 0.0, 'rel_f1_score_addresses': 0.0, 'rel_true_positives_addresses': 0, 'rel_false_positives_addresses': 0, 'rel_false_negatives_addresses': 0,
        }
        # Only fill columns that actually exist in the merged dataframe
        cols_to_fill = {k: v for k, v in fill_values.items() if k in results_df.columns}
        # Ensure consistency in data types before filling
        for col, default_val in cols_to_fill.items():
             if isinstance(default_val, bool):
                 results_df[col] = results_df[col].astype('boolean') # Use nullable boolean
             elif isinstance(default_val, int):
                 # Use float if NaNs might exist, otherwise Int64 for nullable integer
                 if results_df[col].isnull().any():
                      results_df[col] = results_df[col].astype(pd.Int64Dtype()) # Nullable Integer
                 else:
                      results_df[col] = results_df[col].astype(int)
             elif isinstance(default_val, float):
                 results_df[col] = results_df[col].astype(float)

        results_df.fillna(value=cols_to_fill, inplace=True)

        # Ensure boolean type for is_valid and ontology_conforms after filling NaNs
        if 'is_valid' in results_df.columns:
             results_df['is_valid'] = results_df['is_valid'].astype(bool)
        if 'ontology_conforms' in results_df.columns:
             results_df['ontology_conforms'] = results_df['ontology_conforms'].astype(bool)


        print(f"Expanded results to {len(results_df)} expected files (including missing).")

    elif total_articles == 0:
         print("No article directories found. Cannot calculate expected results.")
    else: # results_df is empty but total_articles > 0
         print("Warning: No result files were found or processed. Cannot calculate statistics.")
         # Optionally create an empty expected_df structure if needed downstream
    # --- End Modification ---

    # Calculate summary statistics (Now based on the complete DataFrame)
    if len(results_df) > 0 and 'is_valid' in results_df.columns: # Check if df is usable
        if "model" in results_df.columns:
            print("\nResults by model:")
            # Corrected aggregation syntax for SeriesGroupBy
            model_stats = results_df.groupby("model")["is_valid"].agg(
                Valid='sum',
                Total_Expected='size', # size gives the total rows in the group (expected files)
                Valid_Rate=lambda x: x.mean() * 100 # mean of boolean gives the rate
            ).reset_index() # Add reset_index() to make 'model' a column again
            print(model_stats)

        if "strategy" in results_df.columns:
            print("\nResults by strategy:")
            # Corrected aggregation syntax for SeriesGroupBy
            strategy_stats = results_df.groupby("strategy")["is_valid"].agg(
                Valid='sum',
                Total_Expected='size',
                Valid_Rate=lambda x: x.mean() * 100
            ).reset_index() # Add reset_index()
            print(strategy_stats)
            
    # Save results to CSV
    if args.output_file:
        output_path = args.output_file
    else:
        output_dir = os.path.join(project_root, "experiment-1", "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Collect problematic files if requested
    if args.collect_problematic:
        problematic_dir = os.path.join(project_root, "experiment-1", "problematic_files", timestamp)
        print(f"\nCollecting problematic files to {problematic_dir}")
        
        # Define collection criteria
        reasons = {
            "invalid": lambda x: not x["is_valid"] if "is_valid" in x else False,
            "low_f1": lambda x: x["is_valid"] and x["f1_score"] < args.f1_threshold if "f1_score" in x else False,
            "high_false_positives": lambda x: x["is_valid"] and x["false_positives"] > x["true_positives"] * 2 
                                   if "false_positives" in x and "true_positives" in x else False
        }
        
        collected = collect_problematic_files(results_df, problematic_dir, reasons)
        if len(collected) > 0:
            print(f"Collected {len(collected)} problematic files:")
            
            for reason in collected["reason"].unique():
                count = len(collected[collected["reason"] == reason])
                print(f"  - {reason}: {count} files")
                
            # Save collection info
            collection_info_path = os.path.join(problematic_dir, "collection_info.csv")
            collected.to_csv(collection_info_path, index=False)
            print(f"Collection info saved to {collection_info_path}")
        else:
            print("No problematic files matched the collection criteria.")
    
    # Generate report if requested
    if args.generate_report:
        report_dir = args.report_dir or os.path.join(project_root, "experiment-1", "reports", f"evaluation_{timestamp}")
        print(f"\nGenerating evaluation report in {report_dir}")
        
        stats = generate_evaluation_report(
            results_df, 
            output_dir=report_dir, 
            include_visualizations=False
        )
        
        # Print summary stats
        print("\nEvaluation Summary:")
        print(f"Total files: {stats['total_files']}")
        print(f"Valid files: {stats['valid_files']} ({stats['valid_percentage']:.1f}%)")
        
        if 'overall_performance' in stats:
            print("\nOverall Performance (valid files only):")
            for metric, value in stats['overall_performance'].items():
                print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated knowledge graphs")
    
    # Basic configuration
    parser.add_argument("--results-dir", default=os.path.join(project_root, "experiment-1", "results"),
                        help="Directory containing results files")
    parser.add_argument("--ground-truth-dir", default=os.path.join(project_root, "test_data"),
                        help="Directory containing ground truth files")
    parser.add_argument("--article-prefix", default="EU_",
                        help="Prefix for article directories")
    
    # Filtering options
    parser.add_argument("--model-filter", default=None, 
                        help="Only process files matching this model name")
    parser.add_argument("--strategy-filter", default=None,
                        help="Only process files matching this strategy name")
    
    # Output options
    parser.add_argument("--output-file", "--output-csv", dest="output_file", default=None,
                        help="Path to save CSV results")
    
    # Problematic file collection
    parser.add_argument("--collect-problematic", action="store_true",
                        help="Collect problematic files for further analysis")
    parser.add_argument("--f1-threshold", type=float, default=0.2,
                        help="F1 score threshold below which files are considered problematic")
    
    # Report generation
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate detailed evaluation report")
    parser.add_argument("--report-dir", default=None,
                        help="Directory to save evaluation report")
    
    # New format arguments
    parser.add_argument("--generated-format", choices=['turtle','json-ld'], default='turtle',
                        help="Format of the generated files (turtle or json-ld).")
    parser.add_argument("--ground-truth-format", choices=['turtle','json-ld'], default='turtle',
                        help="Format of the ground truth files (turtle or json-ld).")

    args = parser.parse_args()
    
    main(args)