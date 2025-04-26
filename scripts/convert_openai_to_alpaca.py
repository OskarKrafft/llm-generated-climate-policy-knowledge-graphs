# filepath: scripts/convert_openai_to_alpaca.py
import json
from pathlib import Path
import argparse

# Add project root to sys.path if needed, or ensure script is run from root
# import sys
# script_dir = Path(__file__).resolve().parent
# project_root = script_dir.parent
# sys.path.append(str(project_root))

def convert_openai_jsonl_to_alpaca_jsonl(input_openai_jsonl, output_alpaca_jsonl):
    """
    Converts an OpenAI-style JSONL file (with 'messages' list)
    to Alpaca-style JSONL file (instruction, input, output).
    """
    # Fixed instruction for the fine-tuning task (matches Phase 3 script)
    alpaca_instruction = (
        "Generate RDF/Turtle using the POLIANNA ontology for the following EU policy text snippet. "
        "Adhere strictly to the output rules: exactly one 'turtle' code fence, valid TTL syntax, "
        "POLIANNA compliance, necessary prefixes, and correct triple termination."
    )
    print(f"Converting '{input_openai_jsonl}' to '{output_alpaca_jsonl}'...")
    count = 0
    skipped = 0

    with open(input_openai_jsonl, 'r') as infile, open(output_alpaca_jsonl, 'w') as outfile:
        for line in infile:
            try:
                openai_data = json.loads(line.strip())
                messages = openai_data.get("messages", [])

                user_content = None
                assistant_content = None

                # Extract user (input text) and assistant (target TTL) content
                # Adapt this logic if your message structure is different
                if len(messages) >= 2:
                     # Simple assumption: first is system/user, last is assistant
                     # More robust: iterate and check roles 'user'/'assistant'
                    for msg in messages:
                        if msg.get("role") == "user":
                            user_content = msg.get("content")
                        elif msg.get("role") == "assistant":
                            assistant_content = msg.get("content")

                if user_content and assistant_content:
                    # Ensure assistant_content is a string
                    if not isinstance(assistant_content, str):
                        print(f"Warning: Skipping line {count+1}. Assistant content is not a string.")
                        skipped += 1
                        continue

                    # Add turtle code fences if they are missing
                    assistant_content_stripped = assistant_content.strip()
                    if not (assistant_content_stripped.startswith("```turtle") and assistant_content_stripped.endswith("```")):
                        print(f"Info: Adding turtle fences to assistant content on line {count+1}.")
                        # Ensure there's a newline after the opening fence if not present
                        if not assistant_content_stripped.startswith("\n"):
                             assistant_content = f"```turtle\n{assistant_content_stripped}\n```"
                        else:
                             assistant_content = f"```turtle{assistant_content_stripped}\n```" # Keep existing leading newline if any

                    alpaca_example = {
                        "instruction": alpaca_instruction,
                        "input": user_content,         # EU legislative text
                        "output": assistant_content    # Ground-truth POLIANNA TTL (now with fences)
                    }
                    outfile.write(json.dumps(alpaca_example) + "\n")
                    count += 1
                else:
                    print(f"Warning: Skipping line {count+1}. Could not find user/assistant pair in messages.")
                    skipped += 1

            except json.JSONDecodeError:
                print(f"Warning: Skipping line {count+1}. Error decoding JSON.")
                skipped += 1
            except Exception as e:
                print(f"Warning: Skipping line {count+1}. Unexpected error: {e}")
                skipped += 1

    print(f"Conversion finished. Processed: {count}, Skipped: {skipped}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OpenAI JSONL to Alpaca JSONL format.")
    # Default paths assume script is run from project root
    parser.add_argument(
        "--train-in", type=str, default="training_data.jsonl",
        help="Input OpenAI training JSONL file path."
    )
    parser.add_argument(
        "--val-in", type=str, default="validation_data.jsonl",
        help="Input OpenAI validation JSONL file path."
    )
    parser.add_argument(
        "--train-out", type=str, default="polianna_train_alpaca.jsonl",
        help="Output Alpaca training JSONL file path."
    )
    parser.add_argument(
        "--val-out", type=str, default="polianna_val_alpaca.jsonl",
        help="Output Alpaca validation JSONL file path."
    )
    args = parser.parse_args()

    project_root = Path().cwd() # Assumes running from project root

    convert_openai_jsonl_to_alpaca_jsonl(
        project_root / args.train_in,
        project_root / args.train_out
    )
    convert_openai_jsonl_to_alpaca_jsonl(
        project_root / args.val_in,
        project_root / args.val_out
    )
    print("Both files converted.")