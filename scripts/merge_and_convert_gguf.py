from unsloth import FastLanguageModel
import torch
import os

# Configuration (ADJUSTED FOR 3B MODEL)
base_model_name = "meta-llama/Llama-3.2-3B-Instruct" # *** MATCHES FINE-TUNING ***
adapter_path = "./polianna_llama3.2_3b_finetuned/final_adapter" # *** MATCHES FINE-TUNING OUTPUT ***
output_gguf_file = "polianna-llama3.2-3b-finetuned.q4_K_M.gguf" # *** UPDATED FILENAME ***
quantization_method = "not_quantized" # Common GGUF quantization (q4_K_M, q5_K_M, q8_0, f16 for none)
max_seq_length = 8192 # Should match fine-tuning max_seq_length

# 1. Load Base Model (can use 16-bit precision for merging if VRAM allows, or 4-bit if necessary)
# For merging a 3B model, loading in higher precision (16-bit) should be fine on your hardware.
print(f"Loading base model: {base_model_name} for merging...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = max_seq_length,
    dtype = None, # Auto, or torch.bfloat16/torch.float16
    load_in_4bit = False, # Load in higher precision for merging
    token = os.environ.get("HF_TOKEN"),
)
print("Base model loaded.")

# 2. Merge the LoRA Adapter
# Unsloth's recommended method is to load the adapter onto the base model
# and then use save_pretrained_gguf which handles merging internally.

# 3. Save as GGUF using Unsloth's utility (Recommended)
print(f"Loading adapter from {adapter_path}...")
# Unsloth method to load adapter onto the base model directly
model.load_adapter(adapter_path) # Check Unsloth documentation for the exact method if this changes
print("Adapter loaded.")

print(f"Saving merged model to GGUF: {output_gguf_file} with quantization {quantization_method}")
# This function performs the merge and quantization
model.save_pretrained_gguf(output_gguf_file, tokenizer, quantization_method=quantization_method)
print("GGUF file saved.")
print("Script finished.")