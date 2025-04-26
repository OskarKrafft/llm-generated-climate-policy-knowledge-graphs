import os
import sys
import random
import shutil
from pathlib import Path

# Add the project root to Python's path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# --- Configuration ---
# !! IMPORTANT: Update this path if your TTL files without fullText are elsewhere !!
turtle_no_fulltext_dir = project_root / "polianna-processed" / "turtle_no_fulltext"
raw_data_base_dir = project_root / "polianna-dataset" / "data" / "03a_processed_to_jsonl"
test_data_dir = project_root / "test_data"
training_output_dir = project_root / "training_data"
validation_output_dir = project_root / "validation_data"

# Articles explicitly excluded (in addition to test_data)
few_shot_ids_to_exclude = {
    "EU_32019L0944_Title_0_Chapter_7_Section_5_Article_63",
    "EU_32012L0027_Title_0_Chapter_5_Section_0_Article_24"
}

validation_set_size = 74 # Target size for validation set
# --- End Configuration ---

def copy_article_files(article_id, source_ttl_dir, source_raw_dir, dest_dir):
    """Copies the necessary files for a single article to the destination."""
    article_dest_dir = dest_dir / article_id
    article_dest_dir.mkdir(parents=True, exist_ok=True)

    ttl_source_path = source_ttl_dir / f"{article_id}.ttl"
    policy_info_source_path = source_raw_dir / article_id / "policy_info.json"
    raw_text_source_path = source_raw_dir / article_id / "Raw_Text.txt"

    files_copied = 0
    errors = []

    # Copy TTL (without fullText)
    if ttl_source_path.exists():
        shutil.copy2(ttl_source_path, article_dest_dir / f"{article_id}.ttl")
        files_copied += 1
    else:
        errors.append(f"Missing TTL: {ttl_source_path}")

    # Copy policy_info.json
    if policy_info_source_path.exists():
        shutil.copy2(policy_info_source_path, article_dest_dir / "policy_info.json")
        files_copied += 1
    else:
        errors.append(f"Missing policy_info: {policy_info_source_path}")

    # Copy Raw_Text.txt
    if raw_text_source_path.exists():
        shutil.copy2(raw_text_source_path, article_dest_dir / "Raw_Text.txt")
        files_copied += 1
    else:
        errors.append(f"Missing Raw_Text: {raw_text_source_path}")

    return files_copied == 3, errors # Return True if all 3 files copied, and any errors

def main():
    print("Starting training/validation data split process...")

    # --- Input Validation ---
    if not turtle_no_fulltext_dir.exists():
        print(f"Error: Source TTL directory not found: {turtle_no_fulltext_dir}")
        print("Please ensure the 'turtle_no_fulltext_dir' variable points to the correct location.")
        sys.exit(1)
    if not raw_data_base_dir.exists():
        print(f"Error: Raw data base directory not found: {raw_data_base_dir}")
        sys.exit(1)
    # --- End Input Validation ---

    # 1. Get all available article IDs from the specified turtle directory
    all_article_ids = {f.stem for f in turtle_no_fulltext_dir.glob("*.ttl")}
    if not all_article_ids:
        print(f"Error: No .ttl files found in {turtle_no_fulltext_dir}")
        sys.exit(1)
    print(f"Found {len(all_article_ids)} total articles in {turtle_no_fulltext_dir.name}")

    # 2. Get article IDs used in the test set
    test_article_ids = set()
    if test_data_dir.exists():
        test_article_ids = {d.name for d in test_data_dir.iterdir() if d.is_dir()}
        print(f"Found {len(test_article_ids)} articles in test set ({test_data_dir.name})")
    else:
        print(f"Warning: Test data directory not found: {test_data_dir}")

    # 3. Combine all IDs to exclude
    excluded_ids = test_article_ids.union(few_shot_ids_to_exclude)
    print(f"Excluding {len(few_shot_ids_to_exclude)} specified few-shot/one-shot articles.")
    print(f"Total articles to exclude: {len(excluded_ids)}")

    # 4. Determine articles available for training/validation
    available_article_ids = list(all_article_ids - excluded_ids)
    print(f"Articles available for training/validation: {len(available_article_ids)}")

    if len(available_article_ids) < validation_set_size:
        print(f"Error: Not enough available articles ({len(available_article_ids)}) to create a validation set of size {validation_set_size}.")
        sys.exit(1)

    # 5. Shuffle and split
    random.shuffle(available_article_ids)
    validation_ids = set(available_article_ids[:validation_set_size])
    training_ids = set(available_article_ids[validation_set_size:])

    print(f"Target validation set size: {validation_set_size}")
    print(f"Actual validation set size: {len(validation_ids)}")
    print(f"Training set size: {len(training_ids)}")

    # 6. Create output directories
    training_output_dir.mkdir(exist_ok=True)
    validation_output_dir.mkdir(exist_ok=True)
    print(f"Ensured output directories exist:")
    print(f"  Training: {training_output_dir}")
    print(f"  Validation: {validation_output_dir}")

    # 7. Copy files for validation set
    print("\nProcessing validation set...")
    val_success_count = 0
    val_error_count = 0
    for article_id in validation_ids:
        success, errors = copy_article_files(
            article_id, turtle_no_fulltext_dir, raw_data_base_dir, validation_output_dir
        )
        if success:
            val_success_count += 1
        else:
            val_error_count += 1
            print(f"  Error processing {article_id}: {', '.join(errors)}")
    print(f"Validation set: {val_success_count} articles processed successfully, {val_error_count} with errors.")

    # 8. Copy files for training set
    print("\nProcessing training set...")
    train_success_count = 0
    train_error_count = 0
    for article_id in training_ids:
        success, errors = copy_article_files(
            article_id, turtle_no_fulltext_dir, raw_data_base_dir, training_output_dir
        )
        if success:
            train_success_count += 1
        else:
            train_error_count += 1
            print(f"  Error processing {article_id}: {', '.join(errors)}")
    print(f"Training set: {train_success_count} articles processed successfully, {train_error_count} with errors.")

    print("\nScript finished.")
    if val_error_count > 0 or train_error_count > 0:
        print("Warning: Some articles could not be fully processed due to missing files.")

if __name__ == "__main__":
    main()