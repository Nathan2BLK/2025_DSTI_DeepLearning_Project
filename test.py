# inference_model.py

import torch
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN = 256  # or your actual max_length
MODEL_ID = "NathanDB/toxic-bert-dsti"  # your HF repo

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Load once, cached by HF
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()

def load_thresholds():
    local_path = "test/label_thresholds_test_tuned.json"

    # 1) Local file exists → use it
    if os.path.exists(local_path):
        with open(local_path, "r") as f:
            thr_dict = json.load(f)
        print("Loaded thresholds from LOCAL file.")
        return np.array([thr_dict[label] for label in LABEL_COLS], dtype=np.float32)

    # 2) Try to load from Hugging Face Hub
    try:
        hf_file = hf_hub_download(
            repo_id=MODEL_ID,
            filename="label_thresholds_test_tuned.json",
            repo_type="model",
        )
        with open(hf_file, "r") as f:
            thr_dict = json.load(f)
        print("Loaded thresholds from Hugging Face Hub.")
        return np.array([thr_dict[label] for label in LABEL_COLS], dtype=np.float32)

    except Exception as e:
        print("WARNING: No thresholds found on HF Hub or locally. Using default 0.5.")
        print("Error detail:", e)
        return np.full(len(LABEL_COLS), 0.5, dtype=np.float32)

THRESHOLDS = load_thresholds()

def predict_toxicity(texts):
    if isinstance(texts, str):
        texts = [texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= THRESHOLDS).astype(int)

    pred_dicts = []
    for i in range(len(texts)):
        pred_dicts.append(
            {LABEL_COLS[j]: int(preds[i, j]) for j in range(len(LABEL_COLS))}
        )

    return probs, preds, pred_dicts

print(predict_toxicity("I will kill you!"))
