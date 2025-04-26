# My Thesis Repository: Generating Knowledge Graphs with LLMs

Welcome to the repository for my Master’s thesis on **generating knowledge graph representations of EU law articles using Large Language Models (LLMs)**. This project integrates data preprocessing, ontology management, knowledge graph generation, inference with multiple LLMs (including local and cloud-based), and evaluation of generated graphs against ground truths.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Data](#data)  
4. [Ontology](#ontology)  
5. [Ground Truth Generation](#ground-truth-generation)  
6. [Inference (LLMs)](#inference-llms)  
7. [Evaluation](#evaluation)  
8. [Experiments & Configuration](#experiments--configuration)  
9. [Installation & Setup](#installation--setup)  
10. [Usage](#usage)  
11. [Contributing](#contributing)  
12. [License](#license)

---

## Project Overview

- **Goal**: To evaluate how effectively LLMs (both cloud-based and local) can generate knowledge graph representations of legal text (EU law articles).
- **Data Source**: The [POLIANNA Dataset](https://github.com/kueddelmaier/POLIANNA)
- **Ontology**: Managed in [Protégé](https://protege.stanford.edu/) and exported in multiple formats (OWL, JSON-LD, TTL). The ontology defines the classes, properties, and relationships we use in knowledge graph generation.
- **LLMs**: 
  - **OpenAI GPT-4** or other GPT-series models
  - **Local LLMs** via [Ollama](https://github.com/jmorganca/ollama) or other frameworks (e.g., llama.cpp-based solutions)

**Key Research Questions**:
1. **How do different LLMs perform on generating accurate knowledge graphs from legal text?**  
2. **What effect does the ontology structure or format have on LLM-based knowledge graph generation?**  
3. **What evaluation metrics best capture the quality of these generated knowledge graphs?**

---

## Repository Structure

my-thesis-repo/
├── data/
│   ├── raw/
│   │   ├── 03b_processed_to_json/
│   │   │   ├── article_1/
│   │   │   │   ├── Raw_Text.txt
│   │   │   │   └── Curated_Annotations.json
│   │   │   ├── article_2/
│   │   │   └── ...
│   └── processed/
│       └── # Outputs of preprocessing or ground-truth generation
│
├── ontology/
│   ├── versions/
│   │   ├── ontology_v1.owl
│   │   ├── ontology_v2.owl
│   │   └── ...
│   ├── scripts/
│   │   ├── export_ontology.py
│   │   └── convert_ontology_format.py
│   └── README.md
│
├── src/
│   ├── data_preprocessing/
│   │   ├── preprocess.py
│   ├── ground_truth_generation/
│   │   ├── generate_ground_truth.py
│   ├── inference/
│   │   ├── base_inference.py
│   │   ├── openai_inference.py
│   │   └── local_inference.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   ├── utils/
│   │   └── io_helpers.py
│   └── __init__.py
│
├── experiments/
│   ├── prompt_templates/
│   │   ├── base_prompt.txt
│   │   └── ...
│   └── experiment_configs/
│       ├── experiment1_config.yaml
│       ├── experiment2_config.yaml
│       └── ...
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_ground_truth_generation.py
│   ├── run_inference.py
│   └── run_evaluation.py
│
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_ground_truth_demo.ipynb
│   ├── 02_model_inference_demo.ipynb
│   └── 03_evaluation_demo.ipynb
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_ground_truth_generation.py
│   ├── test_inference.py
│   └── test_evaluation.py
│
├── config/
│   ├── logging_config.yaml
│   └── pipeline_config.yaml
│
├── docs/
│   ├── thesis_writing_notes.md
│   ├── pipeline_overview.md
│   └── ...
│
├── environment.yml or requirements.txt
├── .gitignore
├── LICENSE (optional)
└── README.md


important notes:
Your fine-tuned model ID is: ft:gpt-4o-mini-2024-07-18:pd-berater-der-ffentlichen-hand:polianna-turtle:BOPrF1gr


Experiment 1 (Turtle vs Turtle):
python scripts/run_evaluation.py --results-dir experiment-1/results --ground-truth-dir test_data --output-file experiment-1/evaluation_results.csv --generated-format turtle --ground-truth-format turtle


Experiment 2 (JSON-LD vs JSON-LD):
python scripts/run_evaluation.py --results-dir experiment-2-jsonld/results --ground-truth-dir experiment-2-jsonld/data/test_data --output-file experiment-2-jsonld/evaluation_results.csv --generated-format json-ld --ground-truth-format json-ld


Experiment 2 (Raw -> Turtle vs Turtle):
python scripts/run_evaluation.py --results-dir experiment-2-raw/results --ground-truth-dir test_data --output-file experiment-2-raw/evaluation_results.csv --generated-format turtle --ground-truth-format turtle