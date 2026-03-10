# CodeXtract – Vision-Language Code Extraction Pipeline

## Overview

CodeXtract is a research pipeline for extracting source code from images using a hybrid **Vision–Language Model (VLM) + Code Reconstruction Model** approach.

The system processes screenshots or photos of code (e.g., VSCode screenshots, camera-captured code images) and reconstructs syntactically valid source code while preserving indentation and structure.

The pipeline was developed as part of an academic research project evaluating **Vision-Language Models for Image-to-Code generation**.

---

# Pipeline Architecture

The system uses a multi-stage pipeline:

Image
↓
Qwen2.5-VL (Vision Model)
↓
Raw Code Extraction
↓
Qwen2.5-Coder (Code Repair Model)
↓
Syntax / AST Validation
↓
Retry Loop (if validation fails)
↓
Evaluation Metrics


---

# Key Features

- Vision-based code extraction from screenshots
- Multi-language support
- Code reconstruction using a specialized coding model
- Syntax validation and retry mechanism
- Automatic dataset splitting
- Metrics logging for experimental evaluation
- Timestamp logging for runtime analysis

---

# Supported Programming Languages

The system supports the following languages:

- Python
- C++
- Java
- JavaScript
- HTML
- CSS

Language detection is automatically inferred from the ground-truth code file extension.

---

# Dataset Structure

The dataset must follow the structure below:

dataset/
├── images/
│ ├── 0001.png
│ ├── 0002.png
│ └── ...
│
└── code/
├── 0001.cpp
├── 0002.py
└── ...


Each image must correspond to a ground-truth code file with the **same filename stem**.

Example:
0001.png → 0001.cpp
0002.png → 0002.py


---

# Evaluation Metrics

The pipeline computes the following metrics:

| Metric | Description |
|------|-------------|
| Exact Match | Exact character match between prediction and ground truth |
| Token Accuracy | Percentage of matching tokens |
| CodeBLEU-like | N-gram similarity of generated code |
| Compilation Success | Whether code compiles or passes syntax checks |
| AST Success | Whether code produces a valid Abstract Syntax Tree |

These metrics help evaluate both **syntactic correctness** and **semantic similarity**.

---

# Requirements

## Python

Python 3.10+

Required packages:


Install:
pip install requests


---

## Ollama

The pipeline uses local models via **Ollama**.

Install Ollama:
https://ollama.com


Verify installation:
ollama list


---

# Required Models

Download the following models:
ollama pull qwen2.5vl
ollama pull qwen2.5-coder:7b


These models are used for:

| Model | Purpose |
|------|---------|
| qwen2.5vl | Image → code extraction |
| qwen2.5-coder | Code reconstruction |

---

# Running the Pipeline

Run the pipeline with the following command:

python thesis_img_to_code_pipeline.py
--dataset_dir dataset
--out_dir outputs
--exp E4
--generate_code
--vision_model qwen2.5vl:latest
--coder_model qwen2.5-coder:7b
--max_retries 2
--timeout_s 900
--ollama_bin "/usr/local/bin/ollama"


---

# Output Structure

The pipeline produces the following outputs:
outputs/
├── metrics/
│ ├── E4_train_metrics.json
│ ├── E4_val_metrics.json
│ └── E4_test_metrics.json
│
├── predictions/
│ ├── E4_train_predictions.jsonl
│ ├── E4_val_predictions.jsonl
│ └── E4_test_predictions.jsonl
│
└── generated_code/
├── train/
├── val/
└── test/


---

# Example Output

Example prediction result:
Processed 0001 | Duration: 158.39s


Example metrics:
[E4] TEST METRICS
Exact Match: 0.0000
Token Accuracy: 0.9622
CodeBLEU-like: 0.9103
Compilation Success: 1.0000
AST Success: 1.0000


---

# Experimental Setup

Dataset size used for evaluation:
100 images


Dataset split:

| Split | Percentage |
|------|------------|
| Train | 75% |
| Validation | 15% |
| Test | 10% |

---

# Limitations

- Vision models may perform **semantic reconstruction** instead of exact transcription.
- Exact Match scores may be low even when generated code is correct.
- Performance depends on image quality and resolution.

---

# Future Improvements

Possible improvements include:

- Multi-Vision voting for higher transcription accuracy
- Image preprocessing (contrast enhancement, resizing)
- Code normalization before metric evaluation
- Support for additional programming languages

---

# Author

Shasank Shah  
Research Project – AI DRIVEN VISION LANGUAGE FRAMEWORK FOR CODE GENERATION<img width="468" height="30" alt="image" src="https://github.com/user-attachments/assets/0b467783-f165-4e6f-8fdf-6239234f2cfb" />


---

# License

This project is intended for academic and research use.
