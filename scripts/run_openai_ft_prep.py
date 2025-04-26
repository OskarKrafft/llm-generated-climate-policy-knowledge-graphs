import os
import sys
import time
import argparse
import json  # Added
import hashlib  # Added
from pathlib import Path
from openai import OpenAI, APIError
from dotenv import load_dotenv  # Import dotenv

# Add the project root to Python's path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# --- Configuration ---
DEFAULT_TRAINING_FILE = project_root / "training_data.jsonl"
DEFAULT_VALIDATION_FILE = project_root / "validation_data.jsonl"
DEFAULT_BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Or "gpt-3.5-turbo", etc.
DEFAULT_MODEL_SUFFIX = "polianna-turtle"  # Choose a descriptive suffix
CACHE_FILE = project_root / ".openai_ft_cache.json"  # Added cache file path
# --- End Configuration ---

# --- Cache Functions --- Added
def load_cache(cache_path: Path) -> dict:
    """Loads the cache file."""
    if cache_path.exists():
        try:
            with cache_path.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file {cache_path} is corrupted. Starting fresh.")
            return {}
    return {}

def save_cache(cache_path: Path, cache_data: dict):
    """Saves data to the cache file."""
    try:
        with cache_path.open("w") as f:
            json.dump(cache_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

def get_file_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(4096)  # Read in chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
# --- End Cache Functions ---

def upload_file_to_openai(client: OpenAI, file_path: Path, purpose: str = "fine-tune", cache: dict = None, cache_path: Path = None):  # Added cache params
    """Uploads a file to OpenAI and returns the file ID, using cache if possible."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    file_path_str = str(file_path.resolve())  # Use absolute path as key
    current_hash = get_file_hash(file_path)

    # Check cache
    if cache and "files" in cache and file_path_str in cache["files"]:
        cached_info = cache["files"][file_path_str]
        if cached_info.get("hash") == current_hash and cached_info.get("id"):
            print(f"Using cached file ID for {file_path.name}: {cached_info['id']}")
            return cached_info['id']  # Assume cached ID is valid if hash matches
        else:
            print(f"File {file_path.name} has changed or cache is incomplete. Re-uploading.")

    print(f"Uploading {file_path.name} ({purpose})...")
    try:
        with file_path.open("rb") as f:
            response = client.files.create(file=f, purpose=purpose)
        print(f"  > Upload successful. File ID: {response.id}")
        # Update cache
        if cache is not None and cache_path:
            if "files" not in cache:
                cache["files"] = {}
            cache["files"][file_path_str] = {"id": response.id, "hash": current_hash}
            save_cache(cache_path, cache)
        return response.id
    except APIError as e:
        print(f"  > OpenAI API Error during upload: {e}")
        return None
    except Exception as e:
        print(f"  > An unexpected error occurred during upload: {e}")
        return None

def create_finetuning_job(client: OpenAI, training_file_id: str, validation_file_id: str, base_model: str, suffix: str):
    """Creates a fine-tuning job on OpenAI."""
    print(f"\nCreating fine-tuning job with model '{base_model}'...")
    print(f"  Training file ID: {training_file_id}")
    print(f"  Validation file ID: {validation_file_id}")
    print(f"  Suffix: '{suffix}'")
    try:
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=base_model,
            suffix=suffix,
        )
        print(f"  > Fine-tuning job created successfully. Job ID: {response.id}")
        return response.id
    except APIError as e:
        print(f"  > OpenAI API Error creating job: {e}")
        return None
    except Exception as e:
        print(f"  > An unexpected error occurred creating job: {e}")
        return None

def monitor_finetuning_job(client: OpenAI, job_id: str):
    """Monitors the fine-tuning job until completion or failure."""
    print(f"\nMonitoring fine-tuning job {job_id}...")
    last_status = None
    last_event_time = 0
    while True:
        try:
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            current_status = job_status.status

            if current_status != last_status:
                print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {current_status}")
                last_status = current_status

            # Print recent events
            events_response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
            for event in reversed(events_response.data):  # Print newest first
                if event.created_at > last_event_time:
                    print(f"    - {time.strftime('%H:%M:%S', time.localtime(event.created_at))}: {event.message}")
                    last_event_time = event.created_at

            if current_status in ["succeeded", "failed", "cancelled"]:
                print(f"\nJob {current_status}.")
                if current_status == "succeeded":
                    fine_tuned_model_id = job_status.fine_tuned_model
                    print(f"  > Fine-tuned model ID: {fine_tuned_model_id}")

                    # --- Added: Retrieve and print results ---
                    result_files = job_status.result_files
                    if result_files:
                        print("\n--- Results ---")
                        for result_file_id in result_files:
                            try:
                                # Retrieve file metadata (optional, but good practice)
                                file_metadata = client.files.retrieve(result_file_id)
                                print(f"  Retrieving results file: {file_metadata.filename} ({result_file_id})")
                                # Retrieve file content
                                file_content_response = client.files.content(result_file_id)
                                # The content is bytes, decode it
                                file_content = file_content_response.read().decode('utf-8')
                                print(f"\n{file_content}") # Print the content (usually CSV)
                            except APIError as e:
                                print(f"  > Error retrieving results file {result_file_id}: {e}")
                            except Exception as e:
                                print(f"  > Unexpected error retrieving results file {result_file_id}: {e}")
                        print("--- End Results ---")
                    else:
                        print("  > No result files found for this job.")
                    # --- End Added ---
                    return fine_tuned_model_id
                else:
                    print(f"  > Error details: {job_status.error}")
                    return None
                break  # Exit loop on completion/failure

            # Wait before checking again
            time.sleep(60)  # Check every 60 seconds

        except APIError as e:
            print(f"  > OpenAI API Error monitoring job: {e}. Retrying...")
            time.sleep(60)
        except Exception as e:
            print(f"  > An unexpected error occurred monitoring job: {e}. Retrying...")
            time.sleep(60)

def main(train_file, val_file, base_model, suffix, api_key=None, force_new=False):  # Added force_new
    """Main function to upload files, create, and monitor the fine-tuning job."""
    print("Starting OpenAI Fine-tuning Process...")

    # Load environment variables from .env file
    dotenv_path = project_root / "secrets_config.env"
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load API key from {dotenv_path}")

    # Load cache
    cache = load_cache(CACHE_FILE)  # Added

    # Initialize OpenAI client
    if not api_key:  # If API key is not provided via argument
        api_key = os.getenv("OPENAI_API_KEY")  # Try getting from environment

    if not api_key:
        print(f"Error: OPENAI_API_KEY not found in environment variables (checked {dotenv_path}) or provided via --api-key argument.")
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    # 1. Upload Files (using cache)
    print("\n--- Step 1: Uploading Files ---")
    training_file_id = upload_file_to_openai(client, Path(train_file), purpose="fine-tune", cache=cache, cache_path=CACHE_FILE)  # Pass cache
    validation_file_id = upload_file_to_openai(client, Path(val_file), purpose="fine-tune", cache=cache, cache_path=CACHE_FILE)  # Pass cache

    if not training_file_id or not validation_file_id:
        print("\nError uploading one or both files. Aborting.")
        sys.exit(1)

    # 2. Check for Existing Job in Cache / Create Fine-tuning Job
    print("\n--- Step 2: Checking for Existing Job / Creating Fine-tuning Job ---")
    job_id = None
    cached_job_info = cache.get("last_job")

    # Check if cache is relevant and user doesn't force a new job
    if not force_new and cached_job_info and \
       cached_job_info.get("training_file_id") == training_file_id and \
       cached_job_info.get("validation_file_id") == validation_file_id and \
       cached_job_info.get("base_model") == base_model and \
       cached_job_info.get("suffix") == suffix and \
       cached_job_info.get("id"):

        print(f"Found potentially relevant cached job ID: {cached_job_info['id']}")
        try:
            # Check the status of the cached job
            job_status = client.fine_tuning.jobs.retrieve(cached_job_info['id'])
            current_status = job_status.status
            print(f"  > Status of cached job: {current_status}")

            if current_status in ["validating_files", "queued", "running"]:
                monitor_choice = input(f"  > Do you want to monitor this existing job '{cached_job_info['id']}'? (y/n): ").lower()
                if monitor_choice == 'y':
                    job_id = cached_job_info['id']
                    print(f"Monitoring existing job: {job_id}")
                else:
                    print("Proceeding to create a new job.")
            elif current_status == "succeeded":
                print(f"  > Cached job already succeeded. Fine-tuned model: {job_status.fine_tuned_model}")
                create_new = input(f"  > Do you want to create a new fine-tuning job anyway? (y/n): ").lower()
                if create_new != 'y':
                    print("Exiting.")
                    sys.exit(0)  # Exit if user doesn't want a new one
            elif current_status in ["failed", "cancelled"]:
                print(f"  > Cached job {current_status}. Proceeding to create a new job.")
            else:  # Should not happen based on OpenAI docs, but handle defensively
                print(f"  > Unknown status '{current_status}'. Proceeding to create a new job.")

        except APIError as e:
            print(f"  > Error retrieving status for cached job {cached_job_info['id']}: {e}. Creating a new job.")
        except Exception as e:
            print(f"  > Unexpected error checking cached job {cached_job_info['id']}: {e}. Creating a new job.")

    # Create a new job if no suitable cached job was found/chosen
    if not job_id:
        job_id = create_finetuning_job(client, training_file_id, validation_file_id, base_model, suffix)

        if not job_id:
            print("\nError creating fine-tuning job. Aborting.")
            sys.exit(1)
        else:
            # Cache the new job ID and its config
            cache["last_job"] = {
                "id": job_id,
                "training_file_id": training_file_id,
                "validation_file_id": validation_file_id,
                "base_model": base_model,
                "suffix": suffix,
                "timestamp": time.time()
            }
            save_cache(CACHE_FILE, cache)
            print(f"New job created: {job_id}. Cached.")

    # 3. Monitor Job
    print("\n--- Step 3: Monitoring Job ---")
    fine_tuned_model_id = monitor_finetuning_job(client, job_id)

    if fine_tuned_model_id:
        print("\n------------------------------------")
        print("Fine-tuning completed successfully!")
        print(f"Your fine-tuned model ID is: {fine_tuned_model_id}")
        print("You can now use this ID for inference.")
        print("------------------------------------")
        # Update cache to reflect final state (optional, good practice)
        if "last_job" in cache and cache["last_job"]["id"] == job_id:
            cache["last_job"]["status"] = "succeeded"
            cache["last_job"]["fine_tuned_model"] = fine_tuned_model_id
            save_cache(CACHE_FILE, cache)
    else:
        print("\n------------------------------------")
        print("Fine-tuning job did not succeed.")
        print("------------------------------------")
        # Update cache to reflect final state (optional)
        if "last_job" in cache and cache["last_job"]["id"] == job_id:
            cache["last_job"]["status"] = "failed"  # Or cancelled
            save_cache(CACHE_FILE, cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload data, create, and monitor an OpenAI fine-tuning job, using caching.")  # Updated description
    parser.add_argument("--train-file", type=str, default=str(DEFAULT_TRAINING_FILE),
                        help=f"Path to the training JSONL file (default: {DEFAULT_TRAINING_FILE})")
    parser.add_argument("--val-file", type=str, default=str(DEFAULT_VALIDATION_FILE),
                        help=f"Path to the validation JSONL file (default: {DEFAULT_VALIDATION_FILE})")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"Base model to fine-tune (e.g., 'gpt-4o-mini', 'gpt-3.5-turbo') (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--suffix", type=str, default=DEFAULT_MODEL_SUFFIX,
                        help=f"Suffix for the fine-tuned model name (default: '{DEFAULT_MODEL_SUFFIX}')")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (optional, defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--force-new", action="store_true",  # Added
                        help="Force creation of a new fine-tuning job, ignoring any cached job ID.")

    args = parser.parse_args()

    main(args.train_file, args.val_file, args.base_model, args.suffix, args.api_key, args.force_new)  # Pass force_new