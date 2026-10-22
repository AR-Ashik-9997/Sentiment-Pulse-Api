import torch
import re
import os
import pickle
import pandas as pd
from pathlib import Path
from app.core.ml_loader import ModelManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
VOCAB_PATH = MODELS_DIR / "vocab_metadata.pkl"
MODEL_PATH = MODELS_DIR / "bilstm_sentiment_model.pth"

if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
        vocab = vocab_data.get("vocab")
        max_len = vocab_data.get("max_len", 100)
else:
    vocab = {}
    max_len = 12

VOCAB_SIZE = len(vocab) if vocab else 1000
EMBEDDING_DIM = 64
HIDDEN_SIZE = 32
OUTPUT_DIM = 2
PAD_IDX = vocab.get("<PAD>", 0) if isinstance(vocab, dict) else 0


# --- Utility Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]


def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        return seq + [vocab["<PAD>"]] * (max_len - len(seq))
    else:
        return seq[:max_len]


# --- Prediction Function ---
def predict_sentiment(text, model, vocab, max_len=max_len):
    model.eval()
    cleaned = clean_text(text)
    seq = text_to_sequence(cleaned, vocab)
    padded = pad_sequence(seq, max_len)
    tensor_input = torch.tensor([padded], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    sentiment = "Positive" if predicted_class.item() == 1 else "Negative"
    return sentiment, confidence.item()


# --- Main Dashboard Service Function ---
def generate_review_dashboard_v3(reviews_list, model, vocab, max_len):
    analyzed_results = []

    for text in reviews_list:
        sentiment, confidence = predict_sentiment(text, model, vocab, max_len)
        lower_text = text.lower()

        emotion_keywords = {
            "Joy": [
                "thrilled",
                "amazing",
                "wonderful",
                "love",
                "best",
                "happy",
                "excited",
                "excitement",
            ],
            "Anger": [
                "terrible",
                "awful",
                "worst",
                "hate",
                "angry",
                "horrible",
                "Frustration",
            ],
            "Disappointment": ["disappointed", "poor", "bad", "useless"],
            "Gratitude": ["helpful", "thanks", "thank", "appreciate"],
        }

        detected_emotions = []
        found_keywords = []
        for emotion, keywords in emotion_keywords.items():
            matched = [kw for kw in keywords if kw in lower_text]
            if matched:
                detected_emotions.append(emotion)
                found_keywords.extend(matched)

        emotion_result = f"{', '.join(detected_emotions) if detected_emotions else 'Neutral'}, {', '.join(set(found_keywords)) if found_keywords else 'None'}"

        base_impact = confidence if sentiment == "Positive" else -confidence

        aspects = {
            "Product Quality": 0.0,
            "Shipping Speed": 0.0,
            "Customer Support": 0.0,
            "Pricing / Value": 0.0,
        }

        if any(
            w in lower_text
            for w in ["quality", "build", "setup", "product", "battery", "performance"]
        ):
            aspects["Product Quality"] = round(base_impact, 2)
        if any(
            w in lower_text
            for w in ["shipping", "delivery", "arrived", "fast", "late", "speed"]
        ):
            aspects["Shipping Speed"] = round(base_impact, 2)
        if any(
            w in lower_text for w in ["support", "service", "help", "helpful", "staff"]
        ):
            aspects["Customer Support"] = round(base_impact, 2)
        if any(
            w in lower_text
            for w in ["price", "cost", "value", "expensive", "cheap", "purchase"]
        ):
            aspects["Pricing / Value"] = round(base_impact, 2)

        active_aspects = {k: v for k, v in aspects.items() if v != 0.0}

        is_high_risk = False
        if (
            sentiment == "Negative"
            and confidence > 0.90
            and "Anger" in detected_emotions
        ):
            priority = "Critical"
            is_high_risk = True
        elif sentiment == "Negative" and (
            confidence > 0.80 or "Frustration" in detected_emotions
        ):
            priority = "High"
            is_high_risk = True
        elif sentiment == "Negative" or confidence < 0.60:
            priority = "Medium"
        else:
            priority = "Low"

        if sentiment == "Positive":
            auto_replay = "Thank you so much for your glowing feedback! We're thrilled to hear that the quality and performance exceeded your expectations. We look forward to serving you again soon!"
        else:
            auto_replay = "We sincerely apologize for the issues you experienced with your recent order. Our support escalation team has been notified and we would like to issue a priority replacement or full refund right away."

        analyzed_results.append(
            {
                "text": text,
                "prediction": sentiment,
                "confidence": round(float(confidence), 4),
                "emotion": emotion_result,
                "aspect_impact": active_aspects,
                "priority_level": priority,
                "high_risk": is_high_risk,
                "auto_replay": auto_replay,
            }
        )

    df_results = pd.DataFrame(analyzed_results)

    total_reviews = len(df_results)
    avg_confidence = (
        round(float(df_results["confidence"].mean()), 4) if total_reviews > 0 else 0.0
    )
    positive_count = len(df_results[df_results["prediction"] == "Positive"])
    positive_ratio = (
        round((positive_count / total_reviews) * 100, 2) if total_reviews > 0 else 0.0
    )
    high_risk_escalations = int(df_results["high_risk"].sum())

    aspect_breakdown = {
        "Product Quality": 0.0,
        "Shipping Speed": 0.0,
        "Customer Support": 0.0,
        "Pricing / Value": 0.0,
    }

    for aspects in df_results["aspect_impact"]:
        for asp, val in aspects.items():
            aspect_breakdown[asp] += val

    dashboard_summary = {
        "Total Analyzed Reviews": total_reviews,
        "Avg Confidence Score": avg_confidence,
        "Positive Sentiment Ratio": f"{positive_ratio}%",
        "High Risk Escalations": high_risk_escalations,
        "Aspect-Wise Sentiment Breakdown": {
            k: round(v, 2) for k, v in aspect_breakdown.items() if v != 0.0
        },
    }

    return dashboard_summary, df_results
