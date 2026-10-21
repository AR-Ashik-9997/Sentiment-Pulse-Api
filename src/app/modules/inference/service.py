import os
import torch
from fastapi import HTTPException
from app.model_architecture.BiLSTMClassification import BiLSTMClassification
from app.modules.inference.schemas import ReviewRequest, ReviewDashboardResponse
import pandas as pd
from app.modules.inference.text_preprocessing import (
    MODEL_PATH,
    DEVICE,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_SIZE,
    OUTPUT_DIM,
    PAD_IDX,
    vocab,
    max_len,
    generate_review_dashboard_v3,
)


def run_Sentiment_analysis_inference(
    payload: ReviewRequest,
) -> ReviewDashboardResponse:
    if isinstance(payload.texts, str):
        reviews_list = [payload.texts]
    elif isinstance(payload.texts, list):
        reviews_list = payload.texts
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid input format. Expected a string or a list of strings.",
        )

    if not reviews_list:
        raise HTTPException(status_code=400, detail="Review list cannot be empty.")

    model = BiLSTMClassification(
        VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, OUTPUT_DIM, PAD_IDX
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        raise HTTPException(
            status_code=500, detail=f"Model weights file not found at {MODEL_PATH}"
        )

    model.eval()

    try:

        summary, df_results = generate_review_dashboard_v3(
            reviews_list, model, vocab, max_len
        )
        detailed_results = df_results.to_dict(orient="records")

        return ReviewDashboardResponse(
            summary=summary, detailed_results=detailed_results
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during review inference processing: {str(e)}",
        )
