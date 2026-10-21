# Sentiment Pulse API

Sentiment Pulse API is a Python web service that analyzes customer reviews with
a PyTorch bidirectional LSTM (BiLSTM) classifier. The API returns a sentiment
prediction and confidence score for each review, together with keyword-based
emotion and aspect indicators, a priority level, a high-risk flag, and a
suggested reply. It also provides aggregate dashboard metrics for a batch of
reviews.

## Capabilities

- Analyze multiple review texts in one request.
- Return positive or negative sentiment predictions and confidence scores.
- Summarize review volume, positive sentiment ratio, average confidence,
  high-risk escalations, and aspect-level sentiment impact.
- Select CPU or CUDA automatically through PyTorch.
- Provide interactive API documentation through FastAPI's Swagger UI.

Emotion and aspect indicators are determined by keyword matching; they are not
independent model predictions. Suggested replies are predefined templates
selected according to the sentiment result.

## Requirements

- Python 3.10 or later.
- The Python dependencies specified in `requirements.txt`.
- A trained model checkpoint and its matching vocabulary metadata file in
  `src/models/`.

## Installation

Run these commands from the repository root.

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model files

The inference pipeline expects the following files under `src/models/`:

| File | Description |
| --- | --- |
| `bilstm_sentiment_model.pth` | PyTorch state dictionary for the trained BiLSTM classifier. |
| `vocab_metadata.pkl` | Pickled metadata containing the vocabulary, sequence length, and model configuration used to create the checkpoint. |

The checkpoint and metadata must come from the same training run and be
compatible with the architecture in `src/app/model_architecture/`. Only use
trusted `vocab_metadata.pkl` files: unpickling data from an untrusted source can
execute arbitrary code.

## Run locally

Start the development server from the repository root:

```bash
uvicorn app.main:app --app-dir src --reload
```

By default, the API is available at `http://127.0.0.1:8000`.

| URL | Purpose |
| --- | --- |
| `http://127.0.0.1:8000/` | Service status. |
| `http://127.0.0.1:8000/docs` | Interactive Swagger UI. |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI schema. |

The OpenAPI schema uses the configured API prefix. The prefix and application
title are configurable with the `API_V1_STR` and `PROJECT_NAME` environment
variables. Their defaults are `/api/v1` and `Enterprise AI/ML Sentiment Api
Platform`, respectively.

## API reference

### `GET /`

Returns a JSON object with the service status and configured project name.

### `POST /api/v1/inference/review-dashboard`

Submit a list of reviews in the `texts` field:

```json
{
  "texts": [
    "The product quality is excellent and delivery was fast.",
    "The item arrived late and customer support was unhelpful."
  ]
}
```

The response has two top-level fields:

- `summary`: aggregate metrics for the submitted batch, including total
  analyzed reviews, average confidence, positive sentiment ratio, high-risk
  escalations, and aspect-wise sentiment breakdown.
- `detailed_results`: one result per review, with the original text,
  prediction, confidence, detected emotion keywords, aspect impacts, priority
  level, high-risk status, and suggested reply.

Use `/docs` while the service is running for the complete interactive request
and response schemas.

## Project layout

```text
sentiment-pulse-api/
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
└── src/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── ml_loader.py
    │   ├── model_architecture/
    │   │   ├── __init__.py
    │   │   └── BiLSTMClassification.py
    │   └── modules/
    │       ├── __init__.py
    │       └── inference/
    │           ├── __init__.py
    │           ├── router.py
    │           ├── schemas.py
    │           ├── service.py
    │           └── text_preprocessing.py
    ├── models/
    │   ├── bilstm_sentiment_model.pth
    │   └── vocab_metadata.pkl
    └── fastapi_app.egg-info/  # Generated package metadata (if present)
```

### Main components

- `src/app/main.py` creates the FastAPI application, configures middleware,
  loads resources during application startup, and registers the inference
  router.
- `src/app/core/config.py` defines application settings.
- `src/app/core/ml_loader.py` locates and loads the vocabulary metadata and
  model resources.
- `src/app/model_architecture/BiLSTMClassification.py` defines the PyTorch
  BiLSTM classifier.
- `src/app/modules/inference/router.py` defines the review-dashboard endpoint.
- `src/app/modules/inference/schemas.py` defines request and response models.
- `src/app/modules/inference/service.py` coordinates model inference and builds
  the API response.
- `src/app/modules/inference/text_preprocessing.py` handles text normalization,
  vocabulary encoding, prediction, and dashboard metric generation.
- `src/models/` contains the model checkpoint and matching vocabulary metadata
  required at runtime.
- `pyproject.toml` contains the build-system and package configuration;
  `requirements.txt` lists the pinned Python dependencies and installs this
  project in editable mode.

Virtual environments, Python caches, build output, local environment files,
and editor files are excluded by `.gitignore`. There are currently no automated
test files in the project.
