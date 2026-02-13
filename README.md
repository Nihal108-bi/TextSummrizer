# TextSummrizer

End-to-end dialogue text summarization project built with Hugging Face Transformers, Datasets, and FastAPI.

This project fine-tunes `google/pegasus-cnn_dailymail` on the SAMSum dialogue dataset, evaluates with ROUGE, and serves inference through an API.

## Project Goal

Convert long conversational dialogue into short, meaningful summaries using a modular MLOps-style pipeline.

## What This Project Does

1. Downloads SAMSum dataset artifacts from a configured source.
2. Transforms raw dialogue-summary pairs into tokenized model-ready features.
3. Fine-tunes a PEGASUS Seq2Seq model.
4. Evaluates model quality using ROUGE metrics.
5. Exposes training and prediction endpoints through FastAPI.

## End-to-End Pipeline

1. `Data Ingestion` (`stage_1_data_ingestion_pipeline.py`)
2. `Data Transformation` (`stage_2_data_transformation_pipeline.py`)
3. `Model Training` (`stage_3_model_trainer_pipeline.py`)
4. `Model Evaluation` (`stage_4_model_evalution.py`)
5. `Prediction` (`predicition_pipeline.py` via API)

`main.py` executes stages 1 to 4 in sequence.

## Modular Structure

```text
TextSummrizer/
|-- app.py
|-- main.py
|-- config/
|   `-- config.yaml
|-- params.yaml
|-- requirements.txt
|-- src/
|   `-- TextSummarizer/
|       |-- components/
|       |   |-- data_ingestion.py
|       |   |-- data_transformation.py
|       |   |-- model_trainer.py
|       |   `-- model_evaluation.py
|       |-- config/
|       |   `-- configuration.py
|       |-- constants/
|       |   `-- __init__.py
|       |-- entity/
|       |   `-- __init__.py
|       |-- logging/
|       |   `-- __init__.py
|       |-- pipeline/
|       |   |-- stage_1_data_ingestion_pipeline.py
|       |   |-- stage_2_data_transformation_pipeline.py
|       |   |-- stage_3_model_trainer_pipeline.py
|       |   |-- stage_4_model_evalution.py
|       |   `-- predicition_pipeline.py
|       `-- utils/
|           `-- common.py
|-- artifacts/
|   |-- data_ingestion/
|   |-- data_transformation/
|   |-- model_trainer/
|   `-- model_evaluation/
|-- logs/
|   `-- continuos_logs.log
`-- research/
    |-- 1data_ingestion.ipynb
    |-- 2data_transformation.ipynb
    |-- 3model_trainer.ipynb
    `-- 4model_evaluation.ipynb
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/TextSummarizer/constants/__init__.py` | Defines config file paths (`config/config.yaml`, `params.yaml`). |
| `src/TextSummarizer/entity/__init__.py` | Dataclasses for each stage config contract. |
| `src/TextSummarizer/config/configuration.py` | Reads YAML and returns strongly-typed stage configs. |
| `src/TextSummarizer/components/data_ingestion.py` | Downloads and extracts dataset zip. |
| `src/TextSummarizer/components/data_transformation.py` | Tokenizes dialogue/summary fields and saves transformed dataset. |
| `src/TextSummarizer/components/model_trainer.py` | Fine-tunes PEGASUS and saves model/tokenizer artifacts. |
| `src/TextSummarizer/components/model_evaluation.py` | Computes ROUGE scores and writes `metrics.csv`. |
| `src/TextSummarizer/pipeline/*.py` | Thin orchestration layer that wires configs to components. |
| `src/TextSummarizer/pipeline/predicition_pipeline.py` | Loads trained model/tokenizer and generates summary. |
| `src/TextSummarizer/utils/common.py` | YAML reader and directory creation helpers. |
| `src/TextSummarizer/logging/__init__.py` | Shared logger setup for file + console logs. |

## Configuration

### `config/config.yaml`

Controls artifact paths, dataset source URL, tokenizer/model checkpoint, and evaluation output paths.

### `params.yaml`

Contains Seq2Seq training arguments (epochs, warmup, batch sizes, eval/save strategy, gradient accumulation, generation flag).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training Pipeline

```bash
python main.py
```

This will create/update folders under `artifacts/`.

## Run API Server

Option 1:

```bash
python app.py
```

Option 2:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Swagger docs: `http://127.0.0.1:8080/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Redirects to Swagger docs. |
| `GET` | `/train` | Triggers full training pipeline (`main.py`). |
| `POST` | `/predict?text=...` | Returns generated summary for input dialogue text. |

Example prediction request:

```bash
curl -X POST "http://127.0.0.1:8080/predict?text=Alice%3A%20Meeting%20at%205%3F%20Bob%3A%20Yes%2C%20see%20you%20there."
```

## Artifacts Produced

| Path | Output |
|---|---|
| `artifacts/data_ingestion/` | Downloaded zip and extracted SAMSum dataset. |
| `artifacts/data_transformation/samsum_dataset/` | Tokenized Hugging Face dataset. |
| `artifacts/model_trainer/pegasus-samsum-model/` | Fine-tuned model weights/config. |
| `artifacts/model_trainer/tokenizer/` | Saved tokenizer files. |
| `artifacts/model_evaluation/metrics.csv` | ROUGE metrics CSV. |

## Logging

Runtime logs are written to:

- `logs/continuos_logs.log`

## Tech Stack

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Evaluate (ROUGE)
- FastAPI + Uvicorn

## Notes

- `Dockerfile` and `setup.py` are currently placeholders.
- Current code uses filenames `predicition_pipeline.py` and `stage_4_model_evalution.py` (spelling kept as in repository).

## License

Apache License 2.0
