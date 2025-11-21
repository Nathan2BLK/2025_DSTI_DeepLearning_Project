"""
Gradio app for multi-label toxicity prediction using a Hugging Face model.
Standalone file: app.py

Two modes supported:
 - Remote inference via Hugging Face Inference API (recommended if you don't want to download the model locally). Set env var HF_API_TOKEN if your model is private.
 - Local loading via transformers.from_pretrained (will download model weights locally). This is used if HF_API_TOKEN is not set and you prefer to download the model.

Usage:
 1) Install requirements:
    pip install -r requirements.txt
   or
    pip install transformers torch gradio numpy pandas huggingface-hub

 2) If your model is private, export your token:
    export HF_API_TOKEN=hf_xxx...   # macOS / Linux
    set HF_API_TOKEN=hf_xxx...      # Windows (PowerShell: $env:HF_API_TOKEN = 'hf_xxx')

 3) Run:
    python app.py

Notes:
 - By default the app will try to use the Hugging Face Inference API (no heavy downloads). If no token is found and you want local loading anyway, the app falls back to downloading the model via transformers.
 - Change MODEL_ID to your HF repo id if needed.

"""

import os
import torch
import numpy as np
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: inference API
from huggingface_hub import InferenceApi, hf_api
import tempfile
from pathlib import Path

# ---- Config ----
MODEL_ID = "NathanDB/toxic-bert-dsti"  # change if needed
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN = 256
# thresholds used to convert probabilities -> binary labels
THRESHOLDS = np.array([0.90, 0.25, 0.90, 0.10, 0.40, 0.15], dtype=np.float32)

# ---- Device ----
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---- Choose mode: remote inference (Inference API) if HF_API_TOKEN present, otherwise local from_pretrained ----
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
USE_REMOTE = True  # prefer remote when token available or model is public

inference_api = None
model = None
tokenizer = None

# Try to initialise remote inference if possible
if USE_REMOTE:
    try:
        if HF_API_TOKEN:
            inference_api = InferenceApi(repo_id=MODEL_ID, token=HF_API_TOKEN)
            print("Using Hugging Face Inference API (private or token provided).")
        else:
            # Try without token (works for public models)
            inference_api = InferenceApi(repo_id=MODEL_ID)
            print("Using Hugging Face Inference API (public model).")
    except Exception as e:
        print("Remote Inference API unavailable or failed to init:", e)
        inference_api = None

# If remote inference is not available, fallback to local loading (this will download the model)
if inference_api is None:
    print("Falling back to local model download via transformers.from_pretrained()")
    # Load tokenizer & model once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    print("Model downloaded and loaded locally.")


# ---- Prediction helpers ----

def predict_toxicity_local(text: str):
    """Run the model locally (downloaded weights). Returns (probs, preds, dict)"""
    if not isinstance(text, str) or text.strip() == "":
        return None

    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = (probs >= THRESHOLDS).astype(int)

    result_dict = {lbl: {"probability": float(round(float(probs[i]), 6)), "predicted": bool(preds[i])} for i, lbl in enumerate(LABEL_COLS)}

    return probs, preds, result_dict


def predict_toxicity_remote(text: str):
    """Call Hugging Face Inference API. Tries to map results back to LABEL_COLS."""
    if inference_api is None:
        raise RuntimeError("Inference API is not initialized")
    # The Inference API for sequence-classification returns a list of {label:..., score:...}
    try:
        response = inference_api(inputs=text)
    except Exception as e:
        return None

    # Example response: [{'label': 'toxic', 'score': 0.95}, ...]
    # or: {'error': '...'}
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Inference API error: {response.get('error')}")

    # Normalize into a dict: label -> score
    label_to_score = {}
    if isinstance(response, list):
        for item in response:
            lab = item.get("label")
            score = float(item.get("score", 0.0))
            # Some models return labels like "LABEL_0"; try to handle that
            label_to_score[lab] = score
    elif isinstance(response, dict):
        # Rare forms: model may return a dict with logits; try to handle
        # If 'scores' key exists and is list of floats
        if "scores" in response and isinstance(response["scores"], list):
            # attempt to map by order
            scores = response["scores"]
            for i, lab in enumerate(LABEL_COLS):
                label_to_score[lab] = float(scores[i]) if i < len(scores) else 0.0
        else:
            # fallback empty
            pass

    # Try to match LABEL_COLS case-insensitively
    probs = np.zeros(len(LABEL_COLS), dtype=float)
    for i, lbl in enumerate(LABEL_COLS):
        # direct match
        if lbl in label_to_score:
            probs[i] = label_to_score[lbl]
            continue
        # case-insensitive match
        for k, v in label_to_score.items():
            if k.lower() == lbl.lower():
                probs[i] = v
                break
        else:
            # maybe the label is e.g. "LABEL_0" — try to order-match if counts same
            pass

    # If we didn't get any scores (all zeros), try to infer from ordering if lengths match
    if probs.sum() == 0 and isinstance(response, list) and len(response) == len(LABEL_COLS):
        for i, item in enumerate(response):
            probs[i] = float(item.get("score", 0.0))

    preds = (probs >= THRESHOLDS).astype(int)
    result_dict = {lbl: {"probability": float(round(float(probs[i]), 6)), "predicted": bool(preds[i])} for i, lbl in enumerate(LABEL_COLS)}

    return probs, preds, result_dict


# Unified wrapper used by Gradio
def predict_toxicity(text: str):
    if not isinstance(text, str) or text.strip() == "":
        empty_df = pd.DataFrame(columns=["label", "probability", "predicted"])
        return empty_df, {}

    if inference_api is not None:
        out = predict_toxicity_remote(text)
        if out is None:
            # fallback to local if remote fails and a local model exists
            if model is not None:
                probs, preds, result_dict = predict_toxicity_local(text)
            else:
                raise RuntimeError("Remote call failed and no local model available")
        else:
            probs, preds, result_dict = out
    else:
        probs, preds, result_dict = predict_toxicity_local(text)

    rows = []
    for i, lbl in enumerate(LABEL_COLS):
        prob = float(probs[i])
        pred = int(preds[i])
        rows.append({"label": lbl, "probability": round(prob, 6), "predicted": pred})

    df = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
    return df, result_dict


# Helper to save CSV (temp)
def save_df_to_csv(df: pd.DataFrame):
    tmpdir = Path(tempfile.gettempdir())
    path = tmpdir / f"toxicity_result_{os.getpid()}.csv"
    df.to_csv(path, index=False)
    return str(path)

# Petite fonction pour créer un HTML simple et stylé
def build_result_html(df: pd.DataFrame, result_dict: dict, text: str):
    """Build a stylized HTML result with thin bars and toxicity summary"""
    style = """
    <style>
      .card { background:linear-gradient(180deg,#0b1220,#0f1724); padding:20px; border-radius:14px; color:#e6eef8; font-family:'Segoe UI',Inter,Arial; }
      .title { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; }
      .title h3 { margin:0; font-size:18px; font-weight:700; }
      .title-desc { color:#9fb0c7; font-size:13px; }
      .badge { padding:8px 12px; border-radius:999px; font-weight:700; font-size:14px; }
      .good { background:#10b981; color:#022; }
      .bad { background:#ef4444; color:#fef2f2; }
      
      .summary-box { background:#071226; padding:12px; border-radius:10px; margin-top:14px; border-left:3px solid #ef4444; }
      .summary-box.clean { border-left-color:#10b981; }
      .summary-text { color:#cfe8ff; font-size:14px; line-height:1.5; }
      .summary-text strong { font-weight:700; color:#10b981; }
      .summary-text .toxic-label { background:#ef4444; color:#fef2f2; padding:2px 6px; border-radius:4px; margin:0 4px; font-weight:700; font-size:13px; }
      
      .row { display:flex; align-items:center; gap:10px; margin-top:10px; }
      .label { width:140px; text-transform:capitalize; font-weight:600; color:#cfe8ff; font-size:13px; }
      .bar-container { display:flex; flex-direction:column; gap:4px; flex:1; }
      .bar-bg { background:#071226; width:100%; border-radius:999px; height:6px; overflow:hidden; box-shadow:inset 0 1px 2px rgba(0,0,0,0.3); position:relative; }
      .bar { height:100%; border-radius:999px; transition: width .6s cubic-bezier(0.34, 1.56, 0.64, 1); background:#06b6d4; }
      .threshold-line { position:absolute; top:0; height:100%; width:2px; background:#ef4444; opacity:0.8; }
      .bar-labels { display:flex; justify-content:space-between; font-size:11px; color:#9fb0c7; }
      .prob { min-width:50px; text-align:right; font-weight:700; color:#cfe8ff; font-size:13px; }
      .predicted-badge { padding:2px 6px; border-radius:4px; font-weight:700; font-size:11px; margin-left:8px; }
      .predicted-true { background:#ef4444; color:#fef2f2; }
      .predicted-false { background:#10b981; color:#022; }
    </style>
    """
    html = style + "<div class='card'>"
    
    # Header avec badge
    any_toxic = any([v["predicted"] for v in result_dict.values()])
    status = "<div class='badge bad'>⚠️ Toxic</div>" if any_toxic else "<div class='badge good'>✅ Clean</div>"
    html += f"<div class='title'><div><h3>Toxicity Analysis</h3><div class='title-desc'>Probability per category</div></div>{status}</div>"
    
    # Summary text
    if any_toxic:
        toxic_categories = [lbl.replace('_', ' ').title() for lbl, v in result_dict.items() if v["predicted"]]
        toxic_str = ", ".join([f"<span class='toxic-label'>{cat}</span>" for cat in toxic_categories])
        html += f"<div class='summary-box'><div class='summary-text'><strong>Message detected as toxic</strong> — we identified the following categories: {toxic_str}</div></div>"
    else:
        html += "<div class='summary-box clean'><div class='summary-text'><strong>✅ No toxicity detected</strong> — this message appears safe and appropriate.</div></div>"
    
    # Thin bars with threshold indicators
    html += "<div style='margin-top:16px;'>"
    for i, row in df.iterrows():
        label = row['label']
        label_display = label.replace('_', ' ')
        prob = float(row['probability'])
        is_predicted = result_dict[label]["predicted"]
        threshold = float(THRESHOLDS[LABEL_COLS.index(label)])
        threshold_percent = threshold * 100
        prob_percent = prob * 100
        
        # Badge to show if predicted toxic
        badge_class = "predicted-true" if is_predicted else "predicted-false"
        badge_text = "🚨 Toxic" if is_predicted else "✓ Safe"
        
        html += "<div class='row'>"
        html += f"<div class='label'>{label_display}</div>"
        html += "<div style='display:flex; align-items:center; gap:8px; flex:1;'>"
        html += "<div class='bar-container'>"
        html += "<div class='bar-bg' style='position:relative;'>"
        html += f"<div class='bar' style='width:{prob_percent:.2f}%;'></div>"
        html += f"<div class='threshold-line' style='left:{threshold_percent:.2f}%;'></div>"
        html += "</div>"
        html += f"<div class='bar-labels'><span>0%</span><span style='text-align:center; flex:1;'>Threshold: {threshold_percent:.1f}%</span><span>100%</span></div>"
        html += "</div>"
        html += f"<div class='prob'>{prob_percent:.1f}%</div>"
        html += f"<div class='predicted-badge {badge_class}'>{badge_text}</div>"
        html += "</div></div>"
    html += "</div>"
    html += "</div>"
    return html

# Nouvelle UI Gradio (English)
with gr.Blocks(title="Toxicity Analyzer") as demo:
    gr.HTML("<h2 style='margin:8px 0;color:#e6eef8;font-family:Inter,Arial;text-align:center;'>🛡️ Toxicity Analyzer</h2>")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="Text to analyze", placeholder="Type or paste text here...", lines=6)
            with gr.Row():
                btn = gr.Button("Analyze", variant="primary", scale=2)
                btn_clear = gr.Button("Clear", scale=1)

    with gr.Row():
        out_html = gr.HTML()

    download_file = gr.File(label="📥 Download CSV", visible=False)

    def analyze(text):
        df, result_dict = predict_toxicity(text)
        html = build_result_html(df, result_dict, text)
        csv_path = save_df_to_csv(df)
        return html, csv_path

    def clear_all():
        return "", gr.update(visible=False)

    btn.click(analyze, inputs=txt, outputs=[out_html, download_file])
    btn_clear.click(clear_all, inputs=None, outputs=[txt, download_file])

    gr.Examples(examples=[
        "I will kill you!",
        "You are wonderful and helpful.",
        "Get out of here, you idiot.",
        "This is the best day ever!",
        "I hate everything about this.",
        "You are so stupid and worthless.",
        "Let's grab coffee tomorrow.",
        "Go die in a fire.",
        "Have a great day!",
        "I'm going to punch you in the face."
    ], inputs=txt)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)