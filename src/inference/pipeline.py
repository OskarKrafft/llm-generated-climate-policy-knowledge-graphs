import os
import json
import time
from datetime import datetime
from pathlib import Path
from rdflib import Graph  # added import of Graph for reconstruction checks

from .base_inference import OpenAIClient, OllamaClient
from .prompt_templates import get_prompt_strategy
from .output_parser import parse_output, reconstruct_graph_from_raw  # added reconstruct_graph_from_raw

class KnowledgeGraphPipeline:
    """Pipeline to extract knowledge graphs from policy text using LLMs"""
    
    def __init__(self, llm_client=None, output_dir=None, ontology_path="oontology/ontology_v17_no_fulltext.ttl"):
        """
        Initialize the pipeline with an LLM client.
        
        Args:
            llm_client: Instance of LLMClient (OpenAIClient, OllamaClient, etc.)
            output_dir: Directory to save results
            ontology_path: Path to the TTL file containing the ontology
        """
        self.llm_client = llm_client
        self.ontology_path = ontology_path
        
        # Set default output directory if not provided
        if not output_dir:
            project_root = Path(__file__).resolve().parents[2]
            self.output_dir = project_root / "experiment-1" / "results"
        else:
            self.output_dir = Path(output_dir)
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_article(self, article_text, policy_info, 
                       prompt_strategy='zero-shot', output_format='ttl',
                       max_tokens=2000, temperature=0.0, save_results=True,
                       prompt_dir=None): # Add prompt_dir
        """
        Harmonized function to process a single article and generate a knowledge graph.
        """
        if not self.llm_client:
            raise ValueError("No LLM client provided. Initialize with llm_client parameter.")
        
        # Prepare conversation based on prompt strategy
        conversation = get_prompt_strategy(
            prompt_strategy,
            ontology_path=self.ontology_path,
            article_text=article_text,
            policy_info=json.dumps(policy_info, indent=2),
            output_format=output_format,
            prompt_dir=prompt_dir # Pass prompt_dir
        )
        
        # Generate from LLM
        start_time = time.time()
        llm_output = self.llm_client.generate(
            conversation, 
            max_tokens=max_tokens,
            temperature=temperature
        )

        # FIX: Handle both string responses and dictionary responses
        if isinstance(llm_output, dict) and 'output' in llm_output:
            content = llm_output['output']
        else:
            content = llm_output  # Assume it's already the content string

        generation_time = time.time() - start_time
        
        if not content:
            return {
                "success": False,
                "error": "No output from LLM",
                "generation_time": generation_time
            }
        
        # Parse the output
        result = parse_output(content, format_type=output_format)
        result["generation_time"] = generation_time
        result["output"] = content
        result["success"] = result.get("is_valid", False)
        # ensure metadata exists
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"].update({
            "policy_info": policy_info,
            "prompt_strategy": prompt_strategy,
            "output_format": output_format,
            "model": getattr(self.llm_client, "model", "unknown"),
            "timestamp": datetime.now().isoformat()
        })

        # --- BEGIN Reconstruction Logic for 'raw' format ---
        if output_format.lower() == "raw" and result.get("is_valid") and isinstance(result.get("graph"), list):
            print("  Raw JSON output is valid. Attempting reconstruction to RDF graph...")
            raw_triplets_list = result["graph"]
            article_id = policy_info.get("Titel")
            if not article_id:
                print("  Error: Cannot reconstruct graph. 'Titel' missing in policy_info.")
                result["success"] = False
                result["error"] = "Reconstruction failed: Missing 'Titel' in policy_info."
            else:
                reconstructed_graph = reconstruct_graph_from_raw(raw_triplets_list, article_id)
                if reconstructed_graph is not None:
                    print(f"  Reconstruction successful. Found {len(reconstructed_graph)} triples.")
                    result["graph"] = reconstructed_graph
                    result["metadata"]["reconstructed_from_raw"] = True
                    result["success"] = True
                    result["error"] = None
                else:
                    print("  Error: Reconstruction to RDF graph failed.")
                    result["success"] = False
                    result["error"] = (result.get("error", "") + " | Reconstruction failed.")
        # --- END Reconstruction Logic ---

        # Save results if requested
        if save_results:
            self._save_results(result, policy_info)
        
        return result
    
    def batch_process(self, articles, prompt_strategies=None, output_formats=None, 
                      models=None, max_tokens=2000, temperature=0.0,
                      prompt_dir=None): # Add prompt_dir
        """
        Harmonized function to process multiple articles with different combinations of parameters.
        """
        if not prompt_strategies:
            prompt_strategies = ['zero-shot']
        if not output_formats:
            output_formats = ['ttl']
        if not models:
            models = [{'type': 'openai', 'model': 'gpt-4'}]
        
        results = {}
        
        for article_data in articles:
            article_id = article_data.get('policy_info', {}).get('CELEX_Number', 'unknown')
            results[article_id] = {}
            
            for model_config in models:
                model_name = f"{model_config.get('type', 'unknown')}_{model_config.get('model', 'unknown')}"
                results[article_id][model_name] = {}
                
                # Initialize LLM client for this model
                llm_client = self._get_llm_client(model_config)
                self.llm_client = llm_client
                
                for prompt_strategy in prompt_strategies:
                    results[article_id][model_name][prompt_strategy] = {}
                    
                    for output_format in output_formats:
                        print(f"Processing article {article_id} with {model_name}, {prompt_strategy}, {output_format}")
                        
                        result = self.process_article(
                            article_data['article_text'],
                            article_data['policy_info'],
                            prompt_strategy=prompt_strategy,
                            output_format=output_format,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            prompt_dir=prompt_dir # Pass prompt_dir
                        )
                        
                        results[article_id][model_name][prompt_strategy][output_format] = result
        
        # Save summary report
        self._save_batch_summary(results)
        
        return results
    
    def _get_llm_client(self, model_config):
        """Create an LLM client based on the model configuration"""
        model_type = model_config.get('type', 'openai').lower()
        
        if model_type == 'openai':
            return OpenAIClient(
                model=model_config.get('model', 'gpt-4'),
                api_key=model_config.get('api_key')
            )
        elif model_type == 'ollama':
            return OllamaClient(
                model=model_config.get('model', 'llama3'),
                api_url=model_config.get('api_url', 'http://localhost:11434')
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _save_results(self, result, policy_info):
        """Save individual result to disk"""
        celex = policy_info.get('CELEX_Number', 'unknown')
        model = getattr(self.llm_client, "model", "unknown").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Replace logic with determination of ext and content
        final_graph = result.get('graph')
        original_format = result['metadata']['output_format'].lower()
        if isinstance(final_graph, Graph):
            ext = 'ttl'
            try:
                content = final_graph.serialize(format='turtle')
            except:
                content = result.get('extracted_content', result.get('output', ''))
                ext = 'json' if original_format == 'raw' else original_format
        elif original_format == 'raw' and isinstance(final_graph, list):
            ext = 'json'
            content = json.dumps(final_graph, indent=2)
        else:
            content = result.get('extracted_content', result.get('output', ''))
            if original_format == 'ttl': ext='ttl'
            elif original_format in ['json-ld','jsonld']: ext='jsonld'
            elif original_format == 'raw': ext='json'
            else: ext='txt'
        filename = f"{policy_info.get('CELEX_Number','unknown')}_{getattr(self.llm_client,'model','unknown').replace('/','_')}_{result['metadata']['prompt_strategy']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        path = self.output_dir / filename
        with open(path,'w',encoding='utf-8') as f: f.write(content)
        # Save metadata without graph object
        meta = {k:v for k,v in result.items() if k!='graph'}
        if isinstance(final_graph, Graph): meta['triple_count']=len(final_graph)
        elif isinstance(final_graph, list): meta['raw_triplet_count']=len(final_graph)
        metaf = f"{filename.rsplit('.',1)[0]}_metadata.json"
        with open(self.output_dir/metaf,'w',encoding='utf-8') as mf: json.dump(meta,mf,indent=2,default=str)
        print(f"  Saved output to {filename} and metadata to {metaf}")
        return path, self.output_dir/metaf
    
    def _save_batch_summary(self, results):
        """Save summary of batch processing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"batch_summary_{timestamp}.json"
        
        # Create a simplified summary with just success/failure status and timing
        summary = {}
        
        for article_id, article_results in results.items():
            summary[article_id] = {}
            
            for model_name, model_results in article_results.items():
                summary[article_id][model_name] = {}
                
                for prompt_strategy, strategy_results in model_results.items():
                    summary[article_id][model_name][prompt_strategy] = {}
                    
                    for output_format, result in strategy_results.items():
                        summary[article_id][model_name][prompt_strategy][output_format] = {
                            "success": result.get("success", False),
                            "generation_time": result.get("generation_time"),
                            "triples_count": len(result.get("graph", [])) if result.get("graph") else 0
                        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
