# inference_model.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# If you saved thresholds on HF too, e.g. label_thresholds.json
# you can load them with huggingface_hub or hardcode them here:
THRESHOLDS = np.array([0.90, 0.25, 0.90, 0.10, 0.40, 0.15], dtype=np.float32)
# or just use 0.5 if you prefer

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

print(predict_toxicity("You are an horrible person!"))
