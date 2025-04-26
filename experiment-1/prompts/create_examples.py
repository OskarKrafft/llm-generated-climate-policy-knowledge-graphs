import json
import os
from pathlib import Path

def create_prompt(policy_info, article_text):
    """Create a formatted user prompt from policy info and article text"""
    return (
        f'Below is the **policy metadata** in JSON form:\n{json.dumps(policy_info, indent=2)}\n\n'
        f'Below is the **articles** in raw text:\n{article_text}\n\n'
        f'**Task**: Convert this article into a valid **RDF/Turtle** representation using the POLIANNA ontology, as specified in the system instructions.'
    )

def load_example_data(folder_path):
    """Load all necessary data from an example folder"""
    folder = Path(folder_path)
    
    # Extract article ID from folder name
    article_id = folder.name
    
    # Load policy info
    with open(folder / "policy_info.json", "r", encoding="utf-8") as f:
        policy_info = json.load(f)
    
    # Load article text
    with open(folder / "Raw_Text.txt", "r", encoding="utf-8") as f:
        article_text = f.read().strip()
    
    # Load TTL content - naming might be article_id.ttl or a different pattern
    ttl_files = list(folder.glob("*.ttl"))
    if not ttl_files:
        raise FileNotFoundError(f"No TTL file found in {folder}")
    
    with open(ttl_files[0], "r", encoding="utf-8") as f:
        ttl_content = f.read().strip()
    
    # Create prompt and response
    user_prompt = create_prompt(policy_info, article_text)
    assistant_response = f"```turtle\n{ttl_content}\n```"
    
    return {"user": user_prompt, "assistant": assistant_response}

def main():
    # Set paths
    base_dir = Path(__file__).parent
    one_shot_dir = base_dir / "one_shot_data"
    few_shot_dir = base_dir / "few_shot_data"
    output_one_shot = base_dir / "example_one_shot.txt"
    output_few_shot = base_dir / "examples_few_shot.txt"
    
    # Load one-shot example
    one_shot_folders = [f for f in one_shot_dir.iterdir() if f.is_dir()]
    if not one_shot_folders:
        raise FileNotFoundError(f"No example folders found in {one_shot_dir}")
    
    one_shot_example = load_example_data(one_shot_folders[0])
    
    # Save one-shot example
    with open(output_one_shot, "w", encoding="utf-8") as f:
        json.dump(one_shot_example, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully created {output_one_shot}")
    
    # Load few-shot examples
    examples = [one_shot_example]  # Include the one-shot example first
    
    few_shot_folders = [f for f in few_shot_dir.iterdir() if f.is_dir()]
    for folder in few_shot_folders:
        try:
            example = load_example_data(folder)
            examples.append(example)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    # Save few-shot examples
    with open(output_few_shot, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully created {output_few_shot} with {len(examples)} examples")
    
    # Test if the files load correctly
    test_file("example_one_shot.txt")
    test_file("examples_few_shot.txt")

def test_file(filename):
    """Test if a file loads as valid JSON and has the expected structure"""
    try:
        with open(Path(__file__).parent / filename, "r", encoding="utf-8") as f:
            content = f.read()
        
        data = json.loads(content)
        
        if filename == "example_one_shot.txt":
            assert "user" in data and "assistant" in data
            assert "```turtle" in data["assistant"]
            print(f"✓ {filename} validates successfully")
        else:
            assert isinstance(data, list)
            assert all("user" in item and "assistant" in item for item in data)
            assert all("```turtle" in item["assistant"] for item in data)
            print(f"✓ {filename} validates successfully with {len(data)} examples")
        
    except Exception as e:
        print(f"✗ {filename} validation failed: {e}")

if __name__ == "__main__":
    main()