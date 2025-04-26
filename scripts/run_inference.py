import argparse
import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to Python's path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from src.inference.pipeline import KnowledgeGraphPipeline
from src.inference.base_inference import OpenAIClient, OllamaClient

# Load environment variables from secrets_config.env
dotenv_path = os.path.join(project_root, "secrets_config.env")
load_dotenv(dotenv_path)

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✓ API key loaded successfully")
else:
    print("✗ Failed to load API key")

def get_model_defaults(model_type, model_name):
    """Return default parameters for specific models"""
    defaults = {
        "openai": {
            "gpt-4o": {"max_tokens": 16384, "temperature": 0.0},
            "gpt-4o-mini-2024-07-18": {"max_tokens": 16384, "temperature": 0.0},
            "o1": {"max_tokens": 4000, "temperature": 0.0}
        },
        "ollama": {
            "llama3": {"max_tokens": 4000, "temperature": 0.1},
            "mixtral": {"max_tokens": 3000, "temperature": 0.2},
            "phi3": {"max_tokens": 2000, "temperature": 0.2}
        }
    }
    
    model_defaults = defaults.get(model_type, {}).get(model_name)
    if not model_defaults:
        # Return conservative defaults if model not in the list
        return {"max_tokens": 2000, "temperature": 0.1}
    return model_defaults

def process_single_article(article_id, test_data_dir, output_dir, model_type, model_name, 
                          strategies, temperature, max_tokens, ontology_path, output_format,
                          prompt_dir): # Add prompt_dir
    """Process a single article with specified parameters"""
    # Set up paths
    article_dir = os.path.join(test_data_dir, article_id)
    article_output_dir = os.path.join(output_dir, article_id)
    os.makedirs(article_output_dir, exist_ok=True)
    
    # Load article data
    policy_info_path = os.path.join(article_dir, "policy_info.json")
    raw_text_path = os.path.join(article_dir, "Raw_Text.txt")
    
    if not os.path.exists(policy_info_path) or not os.path.exists(raw_text_path):
        print(f"Error: Required files not found for article {article_id}")
        return None
    
    with open(policy_info_path, 'r', encoding='utf-8') as f:
        policy_info = json.load(f)
    
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        article_text = f.read()
    
    # Set up LLM client
    if model_type.lower() == "openai":
        llm_client = OpenAIClient(model=model_name)
    elif model_type.lower() == "ollama":
        llm_client = OllamaClient(model=model_name)
    else:
        print(f"Error: Unsupported model type '{model_type}'")
        return None
    
    # Create pipeline
    pipeline = KnowledgeGraphPipeline(
        llm_client=llm_client,
        output_dir=article_output_dir,
        ontology_path=ontology_path
    )
    
    # Process with each strategy
    results = {}
    for strategy in strategies:
        print(f"\nProcessing article {article_id} with {strategy} strategy using {model_type}/{model_name} (Format: {output_format})")
        try:
            start_time = time.time()
            result = pipeline.process_article(
                article_text=article_text,
                policy_info=policy_info,
                prompt_strategy=strategy,
                output_format=output_format,
                max_tokens=max_tokens,
                temperature=temperature,
                save_results=True,
                prompt_dir=prompt_dir # Pass prompt_dir
            )
            total_time = time.time() - start_time
            
            # Append processing metadata
            result["total_processing_time"] = total_time
            results[strategy] = result
            
            # Log results
            graph = result.get('graph')
            triple_count = len(graph) if graph is not None else 0
            print(f"  Success: {result.get('success', False)}")
            print(f"  Generation time: {result.get('generation_time', 0):.2f} s")
            print(f"  Total time: {total_time:.2f} s")
            print(f"  Triples: {triple_count}")
            
        except Exception as e:
            print(f"  Error processing with {strategy} strategy: {str(e)}")
            results[strategy] = {
                'success': False,
                'error': f"Processing error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    # Save summary of results
    summary_path = os.path.join(article_output_dir, f"{article_id}_{model_name.replace('/', '_')}_summary.json")
    
    # Create a JSON-serializable version of the results
    serializable_results = {}
    for strategy, result in results.items():
        serializable_results[strategy] = {k: v for k, v in result.items() if k != 'graph'}
        # Optionally add triple count if not already present
        if 'triple_count' not in serializable_results[strategy] and 'graph' in result:
            graph = result.get('graph')
            triple_count = len(graph) if graph is not None else 0
            serializable_results[strategy]['triple_count'] = triple_count
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    return results

def process_test_dataset(test_data_dir, output_dir, model_type, model_name, 
                         strategies, temperature, max_tokens, ontology_path,
                         output_format, prompt_dir, limit=None): # Add prompt_dir
    """Process all articles in the test dataset"""
    # Get all article directories that start with EU_
    article_dirs = [d for d in os.listdir(test_data_dir) 
                    if os.path.isdir(os.path.join(test_data_dir, d)) and d.startswith("EU_")]
    
    if limit and limit > 0:
        article_dirs = article_dirs[:limit]
    
    print(f"Found {len(article_dirs)} articles to process")
    
    all_results = {}
    for article_id in article_dirs:
        results = process_single_article(
            article_id, test_data_dir, output_dir, model_type, model_name,
            strategies, temperature, max_tokens, ontology_path, output_format,
            prompt_dir # Pass prompt_dir
        )
        if results:
            all_results[article_id] = results
    
    # Create a JSON-serializable version for the summary
    serializable_results = {}
    for article_id, article_results in all_results.items():
        serializable_results[article_id] = {}
        for strategy, result in article_results.items():
            serializable_results[article_id][strategy] = {k: v for k, v in result.items() if k != 'graph'}
            # Add triple count if not already present
            if 'triple_count' not in serializable_results[article_id][strategy] and 'graph' in result:
                graph = result.get('graph')
                triple_count = len(graph) if graph is not None else 0
                serializable_results[article_id][strategy]['triple_count'] = triple_count
    
    # Save overall summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"batch_summary_{model_name.replace('/', '_')}_{output_format}_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run LLM knowledge graph generation experiments")
    
    # Required parameters (marked as not required initially to allow config file to provide them)
    parser.add_argument("--test_data", type=str, required=False, 
                        help="Path to test data directory")
    parser.add_argument("--output_dir", type=str, required=False, 
                        help="Path to output directory")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="openai", 
                        choices=["openai", "ollama"], 
                        help="Model provider (openai or ollama)")
    parser.add_argument("--model_name", type=str, default="gpt-4o", 
                        help="Name of the model to use")
    
    # Experiment parameters
    parser.add_argument("--strategies", type=str, nargs="+", 
                        default=["zero-shot", "one-shot", "few-shot"], 
                        help="Prompting strategies to use")
    parser.add_argument("--output_format", type=str, default="ttl", 
                        choices=["ttl", "json-ld", "jsonld"], 
                        help="Desired output format (ttl or json-ld)")
    parser.add_argument("--article_id", type=str, default=None, 
                        help="ID of specific article to process (optional)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of articles to process (optional)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Temperature for LLM generation")
    parser.add_argument("--max_tokens", type=int, default=10000, 
                        help="Maximum tokens for LLM generation")
    
    # Path configurations
    parser.add_argument("--ontology_path", type=str, 
                        default=os.path.join(project_root, "ontology/ontology_v17_no_fulltext.ttl"), 
                        help="Path to ontology file")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to JSON config file (overrides command line args)")
    
    args = parser.parse_args()
    
    # Load from config file if specified
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            # Update args with config values, ensuring command line args take precedence if provided
            for key, value in config.items():
                if getattr(args, key, None) == parser.get_default(key) or getattr(args, key, None) is None:
                     setattr(args, key, value)
                elif key == "strategies" and args.strategies == parser.get_default("strategies"):
                     setattr(args, key, value)

    # Normalize jsonld to json-ld for consistency
    if args.output_format == "jsonld":
        args.output_format = "json-ld"

    # Validate required arguments after potentially loading from config
    if not args.test_data:
        parser.error("the --test_data argument is required (either via command line or config file)")
    if not args.output_dir:
        parser.error("the --output_dir argument is required (either via command line or config file)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine prompt directory based on output directory
    # Assumes output_dir is like 'experiment-X/results/...' or 'experiment-X/results'
    # We want 'experiment-X/prompts'
    output_path = Path(args.output_dir).resolve()
    # Go up until we find a directory likely containing 'prompts'
    # This is a heuristic: assumes 'results' is a subdirectory of the main experiment dir
    experiment_base_dir = output_path
    while experiment_base_dir.name != 'results' and experiment_base_dir.parent != experiment_base_dir:
        experiment_base_dir = experiment_base_dir.parent
    if experiment_base_dir.name == 'results':
        experiment_base_dir = experiment_base_dir.parent # Go up one more level
    
    # Default prompt dir if inference fails
    default_prompt_dir = Path(project_root) / "experiment-1" / "prompts"
    prompt_dir = experiment_base_dir / "prompts"
    if not prompt_dir.is_dir():
        print(f"Warning: Prompt directory not found at {prompt_dir}. Falling back to {default_prompt_dir}")
        prompt_dir = default_prompt_dir
    else:
        print(f"Using prompts from: {prompt_dir}")

    # Get model defaults
    model_defaults = get_model_defaults(args.model_type, args.model_name)

    # Use defaults only if values weren't explicitly provided
    if args.max_tokens is None:
        args.max_tokens = model_defaults["max_tokens"]
    if args.temperature is None:
        args.temperature = model_defaults["temperature"]
    
    # Process article(s)
    if args.article_id:
        # Process single article
        print(f"Processing single article: {args.article_id}")
        process_single_article(
            args.article_id, args.test_data, args.output_dir, 
            args.model_type, args.model_name, args.strategies,
            args.temperature, args.max_tokens, args.ontology_path,
            args.output_format,
            str(prompt_dir) # Pass prompt_dir
        )
    else:
        # Process all articles in test data
        print(f"Processing test dataset with limit={args.limit}")
        process_test_dataset(
            args.test_data, args.output_dir, args.model_type, args.model_name,
            args.strategies, args.temperature, args.max_tokens, args.ontology_path,
            args.output_format,
            str(prompt_dir), # Pass prompt_dir
            args.limit
        )

if __name__ == "__main__":
    main()