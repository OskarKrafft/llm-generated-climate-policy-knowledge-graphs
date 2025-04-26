# From Text to Triples: Evaluating Large Language Model Capabilities for EU Climate Legislation Knowledge Graph Creation

Welcome to the repository for my Master’s thesis. This project focuses on evaluating the capabilities of Large Language Models (LLMs) to automatically generate ontology-constrained knowledge graphs (KGs) from European Union (EU) climate policy articles. It integrates data preprocessing, ontology management, KG generation using multiple LLMs, and comprehensive evaluation against ground truths.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Repository Structure](#repository-structure)
3.  [Data](#data)
4.  [Ontology](#ontology)
5.  [Installation & Setup](#installation--setup)
6.  [Usage](#usage)
    *   [Ground Truth Generation](#ground-truth-generation)
    *   [Inference (LLMs)](#inference-llms)
    *   [Evaluation](#evaluation)
7.  [Experiments & Configuration](#experiments--configuration)

---

## Project Overview

Analysing and accessing European Union (EU) climate legislation is challenging due to rapid legislative changes and the limitations of traditional text-based databases. Knowledge graphs (KGs), structured using domain-specific ontologies, offer a powerful way to efficiently retrieve and interpret legislative information. However, manual creation of detailed KGs is impractical at scale, highlighting the need for automated solutions. This thesis explores the potential of Large Language Models (LLMs) to automatically generate ontology-constrained KGs from EU climate policy articles.

I developed a domain-specific ontology from the expert-annotated [POLIANNA dataset](https://github.com/kueddelmaier/POLIANNA), aligning closely with EU standards such as the European Legislation Identifier (ELI). Using this ontology, a KG dataset was created to assess multiple open-source LLMs, commercial API-based services, and fine-tuned models, across different prompting strategies and structured output formats. Results demonstrated significant variation in LLM effectiveness. Fine-tuned models exhibited the highest accuracy (F > 0.7), indicating that task-specific adaptation significantly enhances reliability. Capable commercial models (e.g., GPT-4o) achieved moderate performance (F > 0.5) when provided explicit prompting guidance. Conversely, tested open-source models performed poorly, often producing semantically incorrect outputs despite syntactic correctness. Analysis highlighted that the primary challenge in LLM-based KG generation is ensuring semantic accuracy and strict adherence to ontology constraints, rather than syntax generation.

This research presents the first systematic benchmarking of LLM-driven, ontology-constrained KG generation for EU climate legislation. While LLMs show significant promise, their reliable implementation currently requires careful fine-tuning, validation, and human oversight. Recommended actions include strengthening EU semantic infrastructure through expert-driven ontology expansion and cautious exploration of LLM-driven KG pilot projects, ultimately enhancing knowledge management and policy analysis in critical areas like climate governance.

---

## Repository Structure

```
my-thesis-repo/
├── data/                     # Raw and processed data (e.g., POLIANNA subsets)
│   ├── polianna-dataset/     # Original POLIANNA data structure (must be downloaded separately)
│   └── polianna-processed/   # Processed data (e.g., Turtle ground truths)
├── ontology/                 # Ontology files and related scripts
│   ├── ontology_v17_no_fulltext.ttl # Main ontology used
│   └── ...
├── src/                      # Source code
│   ├── data_preprocessing/
│   ├── ground_truth_generation/ # Scripts for creating ground truth KGs
│   ├── inference/             # LLM inference logic and pipelines
│   ├── evaluation/            # KG evaluation scripts
│   └── utils/                 # Utility functions
├── experiments/              # Experiment configurations, prompts, results
│   ├── experiment-1/          # Example: Turtle generation experiment
│   │   ├── configs/           # Configuration files (e.g., model params)
│   │   ├── prompts/           # Prompt templates used
│   │   ├── results/           # Raw LLM outputs and summaries
│   │   └── evaluation_results.csv # Evaluation metrics for this experiment
│   ├── experiment-2-jsonld/   # Example: JSON-LD generation experiment
│   └── experiment-2-raw/      # Example: Raw JSON -> Turtle experiment
├── scripts/                  # High-level scripts to run workflows
│   ├── run_ground_truth_generation.py
│   ├── run_inference.py
│   └── run_evaluation.py
├── notebooks/                # Jupyter notebooks for analysis and exploration
│   └── model_evaluation_analysis.ipynb # Analysis of evaluation results
├── figures/                  # Generated plots and figures
├── test_data/                # Subset of data used for testing/evaluation
├── training_data/            # Data used for fine-tuning
├── validation_data/          # Data used for validation during fine-tuning
├── environment.yml           # Conda environment specification
├── combined_evaluation_results.csv # Aggregated results from all experiments
├── secrets_config.env.template # Template for API keys (copy to secrets_config.env)
├── .gitignore
└── README.md
```

---

## Data

The primary data source is the [POLIANNA Dataset](https://github.com/kueddelmaier/POLIANNA), which contains expert annotations of EU legal articles.

*   Raw data is expected to be structured similarly to the POLIANNA format (see `polianna-dataset/`).
*   Scripts in `src/ground_truth_generation/` and `scripts/run_ground_truth_generation.py` are used to convert the raw annotations into Turtle-formatted ground truth knowledge graphs, stored in `polianna-processed/` or `test_data/`.

---

## Ontology

The project uses a custom ontology developed based on POLIANNA annotations and aligned with standards like ELI. The primary ontology file is located at [`ontology/ontology_v17_no_fulltext.ttl`](ontology/ontology_v17_no_fulltext.ttl). This ontology defines the classes (e.g., `PolicyDocument`, `PolicyArticle`, `Actor`) and properties (e.g., `hasArticle`, `addresses`, `contains_instrument`) used for KG generation.

---

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate llm-kg-thesis
    ```
3.  **Set up API Keys:**
    *   Create `secrets_config.env` and add your OpenAI API key (and any other required keys).

---

## Usage

### Ground Truth Generation

To generate the ground truth Turtle files from the processed POLIANNA JSON data:

```bash
# Default: processes data in polianna-dataset/data/03a_processed_to_jsonl
#          outputs to polianna-processed/turtle
python scripts/run_ground_truth_generation.py

# Specify input and output directories
python scripts/run_ground_truth_generation.py <path/to/input/jsonl/folders> <path/to/output/ttl/files>
```
*(See [`scripts/run_ground_truth_generation.py`](scripts/run_ground_truth_generation.py) and [`src/ground_truth_generation/generate_ground_truth.py`](src/ground_truth_generation/generate_ground_truth.py))*

### Inference (LLMs)

To run LLM inference to generate KGs for articles in the test set:

```bash
# Example: Run GPT-4o using zero-shot and one-shot strategies for all articles in test_data
# Output format is Turtle, results saved in experiment-1/results
python scripts/run_inference.py \
    --test_data test_data \
    --output_dir experiment-1/results \
    --model_type openai \
    --model_name gpt-4o-2024-08-06 \
    --strategies zero-shot one-shot \
    --output_format ttl

# Example: Run using a configuration file for experiment 2 (JSON-LD)
python scripts/run_inference.py --config experiment-2-jsonld/configs/inference_config.json

# Example: Run for a single article
python scripts/run_inference.py \
    --test_data test_data \
    --output_dir experiment-1/results \
    --model_type openai \
    --model_name gpt-4o-mini-2024-07-18 \
    --strategies zero-shot \
    --output_format ttl \
    --article_id EU_32004L0008_Title_0_Chapter_0_Section_0_Article_05
```
*(See [`scripts/run_inference.py`](scripts/run_inference.py) and [`src/inference/pipeline.py`](src/inference/pipeline.py))*

### Evaluation

To evaluate the generated KGs against the ground truth:

```bash
# Example: Evaluate Turtle results from experiment-1 against Turtle ground truth in test_data
python scripts/run_evaluation.py \
    --results-dir experiment-1/results \
    --ground-truth-dir test_data \
    --output-file experiment-1/evaluation_results.csv \
    --generated-format turtle \
    --ground-truth-format turtle

# Example: Evaluate JSON-LD results from experiment-2 against JSON-LD ground truth
python scripts/run_evaluation.py \
    --results-dir experiment-2-jsonld/results \
    --ground-truth-dir experiment-2-jsonld/data/test_data \
    --output-file experiment-2-jsonld/evaluation_results.csv \
    --generated-format json-ld \
    --ground-truth-format json-ld
```
*(See [`scripts/run_evaluation.py`](scripts/run_evaluation.py) and [`src/evaluation/evaluation.py`](src/evaluation/evaluation.py))*

Results can be further analysed using the [`model_evaluation_analysis.ipynb`](model_evaluation_analysis.ipynb) notebook.

---

## Experiments & Configuration

Details about specific experiments, including configurations, prompt templates, and results, can be found in the respective `experiment-*/` directories.

*   **Configuration:** Model parameters, temperature, max tokens, etc., can be set via command-line arguments to `run_inference.py` or defined in JSON configuration files (e.g., `experiment-1/configs/`).
*   **Prompts:** Prompt templates for different strategies (zero-shot, one-shot, few-shot) are located in `experiment-*/prompts/`. The inference pipeline selects the appropriate prompt based on the chosen strategy.

---