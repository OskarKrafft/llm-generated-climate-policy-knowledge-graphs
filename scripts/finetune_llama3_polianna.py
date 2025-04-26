from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import os

# 1. Configuration Parameters (ADJUSTED FOR 3x RTX 6000 & Llama 3.2 3B)
model_name = "meta-llama/Llama-3.2-3B-Instruct" # *** Corrected 3.2 3B Instruct model ***
max_seq_length = 8192  # Keep 8192, should fit easily on 3B with 72GB VRAM
output_dir = "./polianna_llama3.2_3b_finetuned" # *** CHANGED output dir name ***
lora_r = 16            # LoRA rank (sensible default)
lora_alpha = 32        # Usually 2*r
lora_dropout = 0.05    # Regularization
use_gradient_checkpointing = "unsloth" # Crucial for VRAM saving
num_train_epochs = 1   # Start with 1 epoch to gauge time
# Adjust batch size for 3 GPUs (24GB each) - can likely fit much more for 3B
per_device_train_batch_size = 4 # *** INCREASED significantly from 1, monitor VRAM ***
# Adjust accumulation steps for effective batch size
# Effective batch size = num_gpus * per_device_train_batch_size * gradient_accumulation_steps
# Effective batch size = 3 * 4 * 4 = 48 (adjust as needed)
gradient_accumulation_steps = 4 # Kept at 4, adjust if needed
learning_rate = 1e-4   # Common starting point for QLoRA
optim = "paged_adamw_32bit" # Recommended by Unsloth
dataset_train_path = "polianna_train_alpaca.jsonl" # Path to your prepared training data
dataset_val_path = "polianna_val_alpaca.jsonl"   # Path to your prepared validation data

# 2. Load Model and Tokenizer with Unsloth 4-bit QLoRA
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection, or torch.bfloat16 if supported
    load_in_4bit = True,
    token = os.environ.get("HF_TOKEN"), # Use HF_TOKEN environment variable if needed
)
print("Model and tokenizer loaded.")

# 3. Configure PEFT (LoRA) Adapters
print("Configuring LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    lora_alpha = lora_alpha,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], # Standard Llama modules
    lora_dropout = lora_dropout,
    bias = "none",
    use_gradient_checkpointing = use_gradient_checkpointing,
    random_state = 42,
    max_seq_length = max_seq_length,
)
print("LoRA adapters configured.")

# 4. Load and Prepare Dataset
print("Loading dataset...")
# Load both train and validation datasets
train_dataset = load_dataset("json", data_files={"train": dataset_train_path}, split="train")
val_dataset = load_dataset("json", data_files={"validation": dataset_val_path}, split="validation") # Load validation data
print(f"Training dataset loaded with {len(train_dataset)} examples.")
print(f"Validation dataset loaded with {len(val_dataset)} examples.") # Print validation size

# Alpaca prompt format (must match the data preparation)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Add EOS token to ensure the model learns to stop.
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

print("Formatting datasets...")
train_dataset = train_dataset.map(formatting_prompts_func, batched=True,)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True,) # Format validation data too
print("Datasets formatted.")

## 5. Define Training Arguments
print("Defining training arguments...")
training_arguments = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    learning_rate = learning_rate,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10, # Log more often than eval/save is fine
    eval_strategy = "epoch", # Invalid argument
    eval_steps = 6,                 # <<< SET TO STEPS PER EPOCH
    save_strategy = "epoch",        # Save checkpoint every epoch (every 6 steps)
    save_total_limit = 2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    max_grad_norm = 0.3,
    warmup_ratio = 0.03,
    lr_scheduler_type = "linear",
    seed = 42,
    # report_to = "wandb",
)
print("Training arguments defined.")

# 6. Initialize Trainer
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,        # Pass validation dataset here
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_arguments,
    packing = False, # Packing might be less stable for complex structured output
)
print("Trainer initialized.")

# 7. Start Fine-tuning
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning finished.")

# 8. Save the final LoRA adapter
final_adapter_path = os.path.join(output_dir, "final_adapter")
print(f"Saving final LoRA adapter to {final_adapter_path}...")
model.save_pretrained(final_adapter_path)
print("Adapter saved.")
print("Script finished.")