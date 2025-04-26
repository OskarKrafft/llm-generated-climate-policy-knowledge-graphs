import json
import os
from pathlib import Path

def create_prompt(policy_info, article_text):
    """Create a formatted user prompt from policy info and article text"""
    # Updated task description and code fence label
    return (
        f'Below is the **policy metadata** in JSON form:\n{json.dumps(policy_info, indent=2)}\n\n'
        f'Below is the **articles** in raw text:\n{article_text}\n\n'
        f'**Task**: Convert this article into a valid **JSON array of raw triplets** using the POLIANNA ontology, as specified in the system instructions. Put your complete JSON array in a single code fence labeled \'json\'.'
    )

def load_example_data(folder_path):
    """Load all necessary data from an example folder"""
    folder = Path(folder_path)

    # Extract article ID from folder name
    article_id = folder.name

    # Load policy info
    policy_info_path = folder / "policy_info.json"
    if not policy_info_path.is_file():
        raise FileNotFoundError(f"policy_info.json not found in {folder}")
    with open(policy_info_path, "r", encoding="utf-8") as f:
        policy_info = json.load(f)

    # Load article text
    article_text_path = folder / "Raw_Text.txt"
    if not article_text_path.is_file():
         raise FileNotFoundError(f"Raw_Text.txt not found in {folder}")
    with open(article_text_path, "r", encoding="utf-8") as f:
        article_text = f.read().strip()

    # --- Refined logic to find and load the raw triplet JSON file ---
    all_json_files = list(folder.glob("*.json"))
    # Explicitly exclude policy_info.json (case-insensitive)
    raw_triplet_files = [f for f in all_json_files if f.name.lower() != "policy_info.json"]

    if not raw_triplet_files:
        raise FileNotFoundError(f"No raw triplet JSON file found in {folder} (checked {len(all_json_files)} json files, excluded policy_info.json)")
    elif len(raw_triplet_files) > 1:
        # Warn if multiple candidates found, but proceed with the first
        print(f"  Warning: Found multiple potential raw triplet files in {folder}: {[f.name for f in raw_triplet_files]}. Using the first one: {raw_triplet_files[0].name}")

    raw_triplet_filepath = raw_triplet_files[0]

    with open(raw_triplet_filepath, "r", encoding="utf-8") as f:
        try:
            raw_triplets_data = json.load(f)
            # Add a type check for debugging
            if not isinstance(raw_triplets_data, list):
                 print(f"  Warning: Loaded data from {raw_triplet_filepath.name} is not a list (type: {type(raw_triplets_data)}). Expected a list of triplets.")
            # Stringify the loaded data (should be the list)
            raw_triplets_str = json.dumps(raw_triplets_data, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {raw_triplet_filepath.name}: {e}") from e
        except Exception as e:
             raise RuntimeError(f"Error processing file {raw_triplet_filepath.name}: {e}") from e
    # --- End refined logic ---

    # Create prompt using policy_info and article_text
    user_prompt = create_prompt(policy_info, article_text)

    # Create assistant response using the stringified raw triplets list
    assistant_response = f"```json\n{raw_triplets_str}\n```"

    return {"user": user_prompt, "assistant": assistant_response}

def main():
    # Set paths
    base_dir = Path(__file__).parent
    one_shot_dir = base_dir / "one_shot_data"
    few_shot_dir = base_dir / "few_shot_data"
    # Updated output filenames
    output_one_shot = base_dir / "example_one_shot_raw.txt"
    output_few_shot = base_dir / "examples_few_shot_raw.txt"
    
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
    
    # Test if the files load correctly - updated filenames
    test_file("example_one_shot_raw.txt")
    test_file("examples_few_shot_raw.txt")

def test_file(filename):
    """Test if a file loads as valid JSON and has the expected structure"""
    try:
        filepath = Path(__file__).parent / filename
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        data = json.loads(content) # Load the outer structure {user: ..., assistant: ...} or list

        is_list_format = filename == "examples_few_shot_raw.txt"
        items_to_check = data if is_list_format else [data]
        all_items_valid = True

        for i, item in enumerate(items_to_check):
            item_label = f"item {i} in {filename}" if is_list_format else filename
            try:
                assert "user" in item and "assistant" in item, f"Missing 'user' or 'assistant' key in {item_label}"
                assistant_content = item["assistant"]
                assert isinstance(assistant_content, str), f"'assistant' value is not a string in {item_label}"

                # Robust splitting
                start_marker = "```json\n"
                end_marker = "\n```"
                if start_marker not in assistant_content or end_marker not in assistant_content:
                     raise ValueError(f"Code fence markers not found in {item_label}")

                # Split once from start, then once from the end of that part
                _, _, after_start = assistant_content.partition(start_marker)
                json_content_str, _, _ = after_start.rpartition(end_marker)
                json_content_str = json_content_str.strip() # Strip whitespace

                if not json_content_str:
                    raise ValueError(f"Extracted JSON content is empty in {item_label}")

                parsed_json = json.loads(json_content_str) # Try parsing the extracted content

                assert isinstance(parsed_json, list), f"Parsed JSON content is not a list in {item_label}"

            except Exception as inner_e:
                all_items_valid = False
                print(f"✗ Validation failed for {item_label}: {type(inner_e).__name__}: {inner_e}")
                if isinstance(inner_e, json.JSONDecodeError):
                     print(f"--- Failing JSON content for {item_label} ---")
                     print(json_content_str)
                     print("--- End Failing JSON content ---")

        if all_items_valid:
             count_str = f" with {len(items_to_check)} examples" if is_list_format else ""
             print(f"✓ {filename} validates successfully{count_str}")
        else:
             raise ValueError(f"One or more items in {filename} failed validation.")

    except Exception as e:
        print(f"✗ {filename} validation failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()