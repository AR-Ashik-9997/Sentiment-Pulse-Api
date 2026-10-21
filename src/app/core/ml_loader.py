import os
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from app.model_architecture.BiLSTMClassification import BiLSTMClassification 

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Automatic device detection (Server-e GPU thakle cuda, na thakle cpu hobe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelManager:
    _models = {}
    _vocab = None
    _max_len = None

    @classmethod
    def load_resources(cls):
        if not MODELS_DIR.exists():
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return

        # 1. Metadata Load
        metadata_path = MODELS_DIR / "vocab_metadata.pkl"
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found at {metadata_path}")
            return

        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            cls._vocab = metadata.get("vocab")
            cls._max_len = metadata.get("max_len")
            print(f"Successfully loaded metadata from {metadata_path}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return

        # 2. PyTorch Model Load (Automatic GPU/CPU mapping)
        model_file = MODELS_DIR / "bilstm_sentiment_model.pth"
        if model_file.exists():
            model_name = "bilstm_sentiment_model"
            print(f"Loading AI Model '{model_name}' on {device.type.upper()}...")

            try:
                vocab_size = metadata.get("vocab_size")
                embedding_dim = metadata.get("embedding_dim")
                hidden_size = metadata.get("hidden_size")  # hidden_size use kora holo
                output_dim = metadata.get("output_dim")
                pad_idx = metadata.get("pad_idx", 0)  # pad_idx add kora holo

                # Model instantiate kore device-e pathano (.to(device))
                model = BiLSTMClassification(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size,
                    output_dim=output_dim,
                    pad_idx=pad_idx,
                ).to(device)

                # State dict load korar somoy map_location hishebe device use kora
                state_dict = torch.load(model_file, map_location=device)
                model.load_state_dict(state_dict)

                model.eval()
                cls._models[model_name] = model

                print(f"Successfully loaded model on {device.type.upper()}!")

            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
        else:
            print(f"Warning: Model file not found at {model_file}")

    @classmethod
    def get_model(cls, name: str):
        return cls._models.get(name)

    @classmethod
    def get_vocab(cls):
        return cls._vocab

    @classmethod
    def get_max_len(cls):
        return cls._max_len

    @classmethod
    def get_device(cls):
        return device
