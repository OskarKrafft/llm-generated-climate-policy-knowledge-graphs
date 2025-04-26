import os
import sys
import json
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

from src.inference.base_inference import OllamaClient
from src.inference.prompt_templates import get_prompt_strategy

def test_ollama_connection(server_url="http://localhost:11434"):
    """Test basic connection to Ollama server"""
    client = OllamaClient(api_url=server_url)
    if client.check_connection():
        print("✅ Successfully connected to Ollama server")
    else:
        print("❌ Failed to connect to Ollama server")
        return False
    return True

def test_basic_generation(model="llama3", server_url="http://localhost:11434"):
    """Test basic text generation"""
    client = OllamaClient(model=model, api_url=server_url)
    
    prompt = "Summarize the key concepts of knowledge graphs in 3 sentences:"
    print(f"Sending prompt: {prompt}")
    
    response = client.generate_from_prompt(prompt, max_tokens=300)
    
    if response:
        print("\nResponse received:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        return True
    else:
        print("❌ Failed to get response")
        return False

def test_kg_prompt(model="llama3", server_url="http://localhost:11434", strategy="zero-shot"):
    """Test knowledge graph prompt with a simple article"""
    client = OllamaClient(model=model, api_url=server_url)
    
    # Simple test policy
    policy_info = {
        "CELEX_Number": "TEST123",
        "Title": "Test Regulation on Energy Efficiency",
        "ELI_URI": "http://data.europa.eu/eli/reg/test/123"
    }
    
    # Simple article text
    article_text = """
    Article 5
    Member States shall ensure that by 31 December 2025, all building automation and 
    control systems in buildings are capable of monitoring energy consumption.
    The Commission shall establish guidelines for the implementation of this requirement.
    """
    
    # Get the appropriate prompt strategy
    ontology_path = os.path.join(project_root, "ontology", "ontology_v17_no_fulltext.ttl")
    conversation = get_prompt_strategy(
        strategy,
        ontology_path=ontology_path,
        article_text=article_text,
        policy_info=json.dumps(policy_info, indent=2),
        output_format="ttl"
    )
    
    print(f"Testing {strategy} prompt with {model}...")
    start_time = datetime.now()
    
    # Generate the response
    response = client.generate(conversation, max_tokens=4000)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if response:
        print(f"✅ Response received in {duration:.2f} seconds")
        
        # Save the response for inspection
        output_dir = os.path.join(project_root, "experiments", "results", "ollama_tests")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"test_{model}_{strategy}_{timestamp}.ttl")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)
            
        print(f"Response saved to {output_file}")
        return True
    else:
        print("❌ Failed to get response")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ollama integration")
    parser.add_argument("--server", type=str, default="http://localhost:11434", 
                      help="Ollama server URL")
    parser.add_argument("--model", type=str, default="llama3", 
                      help="Model to test with")
    parser.add_argument("--test", type=str, choices=["connection", "basic", "kg", "all"], 
                      default="all", help="Test to run")
    parser.add_argument("--strategy", type=str, choices=["zero-shot", "one-shot", "few-shot"], 
                      default="zero-shot", help="Prompt strategy for KG test")
    
    args = parser.parse_args()
    
    if args.test == "connection" or args.test == "all":
        test_ollama_connection(args.server)
        
    if args.test == "basic" or args.test == "all":
        test_basic_generation(args.model, args.server)
        
    if args.test == "kg" or args.test == "all":
        test_kg_prompt(args.model, args.server, args.strategy)