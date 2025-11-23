"""
Gradio app for multi-label toxicity prediction using a Hugging Face model.
Standalone file: app.py

The model is ALWAYS loaded from the Hugging Face Hub via `from_pretrained(MODEL_ID)`.
No Inference API is used. This works locally and on Hugging Face Spaces.

Usage:
 1) Install requirements:
    pip install transformers torch gradio numpy pandas huggingface-hub

 2) Run locally:
    python app.py

On Hugging Face Spaces:
 - The app logs to /data/toxicity_history.csv (persistent, not in the repo).
"""

import os
import json
import re
import csv
from html import escape
import tempfile
from pathlib import Path
from datetime import datetime
import shutil

import torch
import numpy as np
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ---- Config ----
MODEL_ID = "NathanDB/toxic-bert-dsti"  # change if needed
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN = 128  # max token length for the model

# ---- Admin access (for downloading logs) ----
# On HF Spaces, set ADMIN_KEY in Settings -> Variables & secrets
ADMIN_KEY = os.environ.get("ADMIN_KEY", None)

def admin_get_logs(key: str):
    """
    Gradio callback to let the admin download the log file.
    - Returns the path to a TEMP COPY of LOG_FILE (in /tmp) if key matches ADMIN_KEY.
    - Returns None otherwise (file output will stay empty).
    """
    # If no admin key configured, disable download
    if not ADMIN_KEY:
        print("ADMIN_KEY not set, refusing admin download.")
        return None

    if key != ADMIN_KEY:
        print("Wrong admin key, refusing admin download.")
        return None

    if not LOG_FILE.exists():
        print("Log file does not exist yet.")
        return None

    # Copy to a temp location Gradio is happy with (/tmp)
    tmpdir = Path(tempfile.gettempdir())
    tmp_path = tmpdir / f"toxicity_history_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
    shutil.copy2(LOG_FILE, tmp_path)
    print("Admin download authorized, serving temp copy:", tmp_path)

    return str(tmp_path)

# ---- Human-readable info per category (for the report) ----
LABEL_INFO = {
    "toxic": {
        "name": "Toxic",
        "severity": "medium",
        "desc": "General rude, aggressive, or hostile language that may be harmful to the conversation.",
    },
    "severe_toxic": {
        "name": "Severe toxic",
        "severity": "high",
        "desc": "Extremely aggressive, abusive, or hateful language, often stronger than standard toxicity.",
    },
    "obscene": {
        "name": "Obscene",
        "severity": "medium",
        "desc": "Explicitly vulgar or coarse language, often including swear words and offensive expressions.",
    },
    "threat": {
        "name": "Threat",
        "severity": "critical",
        "desc": "Language that suggests violence, self-harm, or other physical danger directed at someone.",
    },
    "insult": {
        "name": "Insult",
        "severity": "medium",
        "desc": "Direct attacks or derogatory statements about a person or group.",
    },
    "identity_hate": {
        "name": "Identity hate",
        "severity": "critical",
        "desc": "Attacks targeting a person or group based on identity (race, religion, gender, etc.).",
    },
}

# ---- Simple lexical dictionary to highlight risky words in the text ----
TOXIC_KEYWORDS = {
    "idiot": "insult",
    "stupid": "insult",
    "moron": "insult",
    "dumb": "insult",
    "worthless": "insult",
    "loser": "insult",
    "hate": "toxic",
    "kill": "threat",
    "die": "threat",
    "hang": "threat",
    "punch": "threat",
    "slap": "threat",
    "shit": "obscene",
    "fuck": "obscene",
    "fucking": "obscene",
    "bitch": "insult",
    "bastard": "obscene",
    "asshole": "insult",
    "retard": "insult",
    "retarded": "insult",
    # identity / group related terms – contextual, but useful as signal
    "nigger": "identity_hate",
    "faggot": "identity_hate",
    "spic": "identity_hate",
    "chink": "identity_hate",
    "kike": "identity_hate",
}

# ---- Thresholds used to convert probabilities -> binary labels ----
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

# ---- Logging config ----
# HF Spaces: log to /data (persistent, not in repo)
# Local: log to <project_dir>/data (add it to .gitignore)

BASE_DIR = Path(__file__).resolve().parent
ON_SPACE = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))

if ON_SPACE:
    base_log_dir = Path("/data")
else:
    base_log_dir = BASE_DIR / "data"

base_log_dir.mkdir(parents=True, exist_ok=True)
LOG_FILE = base_log_dir / "toxicity_history.csv"

def log_interaction(text: str, result_dict: dict):
    """
    Automatic, private logging of:
      - timestamp_utc
      - raw text
      - predicted_labels (comma-separated)
      - probabilities_by_label_json
    On HF Spaces this goes to /data/toxicity_history.csv (not visible in repo).
    """
    if not isinstance(text, str) or text.strip() == "":
        return

    timestamp = datetime.utcnow().isoformat()

    predicted_labels = [
        lbl for lbl, v in result_dict.items() if v.get("predicted", False)
    ]
    predicted_labels_str = ",".join(predicted_labels)

    probs_by_label = {
        lbl: float(v.get("probability", 0.0)) for lbl, v in result_dict.items()
    }

    file_exists = LOG_FILE.exists()
    print("Writing to:", LOG_FILE)

    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "text",
                    "predicted_labels",
                    "probabilities_by_label_json",
                ]
            )
        writer.writerow(
            [
                timestamp,
                text,
                predicted_labels_str,
                json.dumps(probs_by_label, ensure_ascii=False),
            ]
        )

# ---- Admin log retrieval (used by Gradio callback) ----
def admin_get_logs(key: str):
    """
    Gradio callback to let the admin download the log file.
    - Returns the path to LOG_FILE if key matches ADMIN_KEY.
    - Returns None otherwise (file output will stay empty).
    """
    # If no admin key configured, disable download
    if not ADMIN_KEY:
        print("ADMIN_KEY not set, refusing admin download.")
        return None

    if key != ADMIN_KEY:
        print("Wrong admin key, refusing admin download.")
        return None

    if not LOG_FILE.exists():
        print("Log file does not exist yet.")
        return None

    # Return as string path for gr.File
    return str(LOG_FILE)

# ---- Device & model load (always from HF Hub) ----
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")


# ---- Prediction helpers ----
def predict_toxicity(text: str):
    """Run the model (loaded from HF Hub) and return df + result_dict."""

    if not isinstance(text, str) or text.strip() == "":
        empty_df = pd.DataFrame(columns=["label", "probability", "predicted"])
        return empty_df, {}

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

    result_dict = {
        lbl: {
            "probability": float(round(float(probs[i]), 6)),
            "predicted": bool(preds[i]),
        }
        for i, lbl in enumerate(LABEL_COLS)
    }

    rows = []
    for i, lbl in enumerate(LABEL_COLS):
        prob = float(probs[i])
        pred = int(preds[i])
        rows.append({"label": lbl, "probability": round(prob, 6), "predicted": pred})

    df = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(
        drop=True
    )
    return df, result_dict


# Helper to save CSV (per-request results, separate from history log)
def save_df_to_csv(df: pd.DataFrame):
    tmpdir = Path(tempfile.gettempdir())
    path = tmpdir / f"toxicity_result_{os.getpid()}.csv"
    df.to_csv(path, index=False)
    return str(path)


def highlight_risky_words(text: str, result_dict: dict):
    """
    Simple lexical highlighter:
    - Wrap known toxic words in colored spans.
    - Distinguish between words in a category predicted toxic vs only lexically risky.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "", 0, []

    tokens = text.split()
    highlighted_tokens = []
    hits = []

    for tok in tokens:
        clean = re.sub(r"\W+", "", tok).lower()
        if not clean:
            highlighted_tokens.append(escape(tok))
            continue

        if clean in TOXIC_KEYWORDS:
            label = TOXIC_KEYWORDS[clean]
            is_active = bool(result_dict.get(label, {}).get("predicted", False))
            if is_active:
                span = f"<span class='word-bad' title='Flagged as {label} by model'>{escape(tok)}</span>"
            else:
                span = f"<span class='word-soft' title='Lexically risky word (category: {label})'>{escape(tok)}</span>"
            highlighted_tokens.append(span)
            hits.append((clean, label, is_active))
        else:
            highlighted_tokens.append(escape(tok))

    highlighted_html = " ".join(highlighted_tokens)

    unique_hits = {}
    for w, lbl, active in hits:
        key = (w, lbl)
        if key not in unique_hits:
            unique_hits[key] = active
        else:
            unique_hits[key] = unique_hits[key] or active

    hits_list = [(w, lbl, active) for (w, lbl), active in unique_hits.items()]
    return highlighted_html, len(hits_list), hits_list


def build_result_html(df: pd.DataFrame, result_dict: dict, text: str):
    """Build a stylized HTML report with summary, metrics, and highlighted text."""
    style = """
    <style>
      .card { background:linear-gradient(180deg,#020617,#020617); padding:20px; border-radius:14px; color:#e6eef8; font-family:'Segoe UI',Inter,Arial; }
      .title { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; }
      .title h3 { margin:0; font-size:18px; font-weight:700; }
      .title-desc { color:#9fb0c7; font-size:13px; }
      .badge { padding:8px 12px; border-radius:999px; font-weight:700; font-size:14px; }
      .good { background:#10b981; color:#022; }
      .medium { background:#f97316; color:#1f1300; }
      .bad { background:#ef4444; color:#fef2f2; }
      
      .summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin-top:10px; }
      .pill { background:#02091a; border-radius:10px; padding:8px 10px; font-size:12px; color:#cfe8ff; border:1px solid #0b172a; }
      .pill strong { display:block; font-size:11px; letter-spacing:0.04em; text-transform:uppercase; color:#9fb0c7; margin-bottom:3px; }

      .summary-box { background:#071226; padding:12px; border-radius:10px; margin-top:14px; border-left:3px solid #ef4444; }
      .summary-box.clean { border-left-color:#10b981; }
      .summary-text { color:#cfe8ff; font-size:14px; line-height:1.5; }
      .summary-text strong { font-weight:700; color:#10b981; }
      .summary-text .toxic-label { background:#ef4444; color:#fef2f2; padding:2px 6px; border-radius:4px; margin:0 4px; font-weight:700; font-size:13px; }

      .section-title { font-size:13px; text-transform:uppercase; letter-spacing:0.08em; color:#9fb0c7; margin-top:18px; margin-bottom:6px; }
      .section-sub { font-size:12px; color:#7b8ca6; margin-bottom:6px; }

      .text-box { background:#02091a; border-radius:10px; padding:10px; border:1px solid #0b172a; font-size:13px; color:#e6eef8; line-height:1.6; }
      .word-bad { background:#ef4444; color:#fef2f2; padding:0 3px; border-radius:4px; }
      .word-soft { background:#f97316; color:#1f1300; padding:0 3px; border-radius:4px; }

      .lexicon-legend { font-size:12px; color:#9fb0c7; margin-top:6px; }
      .lex-tag { display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:999px; border:1px solid #1e293b; margin:2px 4px 2px 0; }
      .lex-dot-hard { width:8px; height:8px; border-radius:999px; background:#ef4444; }
      .lex-dot-soft { width:8px; height:8px; border-radius:999px; background:#f97316; }

      .summary-list { font-size:13px; color:#cfe8ff; margin-top:4px; padding-left:18px; }
      .summary-list li { margin-bottom:4px; }

      .row { display:flex; align-items:center; gap:10px; margin-top:10px; }
      .label { width:140px; text-transform:capitalize; font-weight:600; color:#cfe8ff; font-size:13px; }
      .bar-container { display:flex; flex-direction:column; gap:4px; flex:1; }
      .bar-bg { background:#02091a; width:100%; border-radius:999px; height:6px; overflow:hidden; box-shadow:inset 0 1px 2px rgba(0,0,0,0.3); position:relative; }
      .bar { height:100%; border-radius:999px; transition: width .6s cubic-bezier(0.34, 1.56, 0.64, 1); background:#06b6d4; }
      .threshold-line { position:absolute; top:0; height:100%; width:2px; background:#ef4444; opacity:0.8; }
      .bar-labels { display:flex; justify-content:space-between; font-size:11px; color:#9fb0c7; }
      .prob { min-width:60px; text-align:right; font-weight:700; color:#cfe8ff; font-size:13px; }
      .margin { min-width:70px; text-align:right; font-weight:500; font-size:11px; color:#9fb0c7; }
      .predicted-badge { padding:2px 6px; border-radius:4px; font-weight:700; font-size:11px; margin-left:8px; }
      .predicted-true { background:#ef4444; color:#fef2f2; }
      .predicted-false { background:#10b981; color:#022; }

      .desc-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:8px; margin-top:6px; }
      .desc-card { background:#02091a; border-radius:10px; padding:8px 10px; border:1px solid #0b172a; font-size:12px; color:#cfe8ff; }
      .desc-title { font-weight:600; margin-bottom:3px; }
      .desc-severity { font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#9fb0c7; margin-bottom:3px; }

      .footer { margin-top:14px; font-size:11px; color:#64748b; display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; }
      .footer a { color:#38bdf8; text-decoration:none; }
      .footer a:hover { text-decoration:underline; }
    </style>
    """

    # No data case
    if df is None or df.empty or not result_dict:
        return style + """
        <div class='card'>
          <div class='title'>
            <div>
              <h3>Toxicity Analysis</h3>
              <div class='title-desc'>Enter a sentence to get a detailed report.</div>
            </div>
            <div class='badge good'>No input</div>
          </div>
          <div class='summary-box clean'>
            <div class='summary-text'>
              Type or paste a message on the left, then click <strong>Analyze</strong>.
            </div>
          </div>
        </div>
        """

    # --- Basic metrics ---
    any_toxic = any(v["predicted"] for v in result_dict.values())
    probs_arr = df["probability"].values.astype(float)
    max_prob = float(probs_arr.max())
    max_label = df.iloc[0]["label"]
    avg_prob = float(probs_arr.mean())

    clean_text = text.strip()
    n_chars = len(clean_text)
    n_tokens = len(clean_text.split()) if clean_text else 0

    # Strong/weak positives, borderline labels, and margin-based confidence
    strong_pos = []
    weak_pos = []
    borderline = []
    margin_abs_list = []

    for lbl, v in result_dict.items():
        prob = float(v["probability"])
        thr = float(THRESHOLDS[LABEL_COLS.index(lbl)])
        margin = prob - thr
        margin_abs = abs(margin)
        margin_abs_list.append(margin_abs)

        if v["predicted"]:
            if margin >= 0.2:
                strong_pos.append((lbl, prob, margin))
            else:
                weak_pos.append((lbl, prob, margin))
        else:
            if thr - 0.05 <= prob < thr:
                borderline.append((lbl, prob, thr - prob))

    global_margin = float(np.mean(margin_abs_list)) if margin_abs_list else 0.0
    if global_margin >= 0.3:
        confidence_label = "High"
    elif global_margin >= 0.15:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    toxic_probs = [v["probability"] for k, v in result_dict.items() if v["predicted"]]
    max_toxic_prob = max(toxic_probs) if toxic_probs else max_prob

    if not any_toxic or max_toxic_prob < 0.20:
        severity_badge_class = "good"
        severity_text = "Low risk"
    elif max_toxic_prob < 0.50:
        severity_badge_class = "medium"
        severity_text = "Medium risk"
    else:
        severity_badge_class = "bad"
        severity_text = "High risk"

    toxic_categories = [
        lbl.replace("_", " ").title() for lbl, v in result_dict.items() if v["predicted"]
    ]

    highlighted_html, n_hits, hits_list = highlight_risky_words(text, result_dict)

    # --- Build HTML ---
    html = style + "<div class='card'>"

    html += (
        "<div class='title'><div><h3>Toxicity Analysis</h3>"
        "<div class='title-desc'>Detailed model report for this message.</div></div>"
        f"<div class='badge {severity_badge_class}'>{severity_text}</div></div>"
    )

    if any_toxic:
        toxic_str = (
            ", ".join(
                [f"<span class='toxic-label'>{cat}</span>" for cat in toxic_categories]
            )
            if toxic_categories
            else "toxic patterns"
        )
        html += (
            "<div class='summary-box'>"
            f"<div class='summary-text'><strong>Message detected as toxic</strong> — "
            f"main categories: {toxic_str}.</div></div>"
        )
    else:
        msg = "this message appears safe and appropriate according to the model and thresholds."
        if n_hits > 0:
            msg = (
                "the model considers this message globally safe, although it contains some potentially risky terms used in a relatively neutral context."
            )
        html += (
            "<div class='summary-box clean'>"
            f"<div class='summary-text'><strong>✅ No toxicity detected</strong> — {msg}</div></div>"
        )

    html += "<div class='summary-grid'>"
    html += (
        "<div class='pill'><strong>Max category</strong>"
        f"{max_label.replace('_',' ').title()} ({max_prob*100:.1f}%)</div>"
    )
    html += "<div class='pill'><strong>Average toxicity</strong>" f"{avg_prob*100:.1f}%</div>"
    html += (
        "<div class='pill'><strong>Model confidence</strong>"
        f"{confidence_label} (margin-based)</div>"
    )
    html += (
        "<div class='pill'><strong>Message length</strong>"
        f"{n_tokens} tokens · {n_chars} chars</div>"
    )
    html += "</div>"

    html += "<div class='section-title'>Interpretation summary</div>"
    html += "<ul class='summary-list'>"

    if any_toxic:
        if strong_pos:
            top_labels = ", ".join(
                [lbl.replace("_", " ").title() for lbl, _, _ in strong_pos]
            )
            html += (
                f"<li>The model finds <strong>strong evidence</strong> of toxicity in: {top_labels}.</li>"
            )
        if weak_pos:
            weak_labels = ", ".join(
                [lbl.replace("_", " ").title() for lbl, _, _ in weak_pos]
            )
            html += (
                f"<li>Some categories are <strong>mildly toxic</strong> (just above their thresholds): {weak_labels}.</li>"
            )
    else:
        html += "<li>The message is globally below all toxicity thresholds.</li>"

    if borderline:
        b_labels = ", ".join(
            [lbl.replace("_", " ").title() for lbl, _, _ in borderline]
        )
        html += (
            f"<li>The following categories are <strong>borderline</strong> (close to the decision threshold): {b_labels}. Small changes in wording could flip the decision.</li>"
        )

    if n_hits > 0:
        html += (
            "<li>The message contains <strong>lexically risky terms</strong> "
            "(highlighted below). Their impact depends on context and tone.</li>"
        )

    if confidence_label == "Low":
        html += (
            "<li>Model confidence is <strong>low</strong> in the sense that many labels lie close to their thresholds, so predictions should be interpreted with extra care.</li>"
        )
    elif confidence_label == "Medium":
        html += (
            "<li>Model confidence is <strong>moderate</strong>: some labels are clearly above/below their thresholds, while others remain near the decision boundary.</li>"
        )
    else:
        html += (
            "<li>Model confidence is <strong>high</strong>: most labels are clearly separated from their thresholds (either strongly toxic or clearly safe).</li>"
        )

    html += "</ul>"

    html += "<div class='section-title'>Message with highlighted risky words</div>"
    html += "<div class='section-sub'>Red = words in categories predicted toxic, orange = lexically risky terms even if the model kept them below threshold.</div>"
    html += "<div class='text-box'>"
    if clean_text:
        html += highlighted_html
    else:
        html += "<span style='color:#64748b;'>No text provided.</span>"
    html += "</div>"

    if n_hits > 0:
        html += "<div class='lexicon-legend'>"
        html += "<span style='margin-right:6px;'>Detected terms:</span>"
        for word, lbl, active in sorted(
            hits_list, key=lambda x: (x[2], x[1], x[0]), reverse=True
        ):
            dot_class = "lex-dot-hard" if active else "lex-dot-soft"
            status_txt = "model: toxic" if active else "lexical only"
            html += (
                f"<span class='lex-tag'><span class='{dot_class}'></span>"
                f"<span>{escape(word)} · {lbl.replace('_',' ')}</span>"
                f"<span style='opacity:0.7;'>({status_txt})</span></span>"
            )
        html += "</div>"

    html += "<div class='section-title'>Per-category probabilities</div>"
    html += "<div class='section-sub'>Bars show the model probability for each label. The vertical red line is the decision threshold used in this app.</div>"
    html += "<div style='margin-top:6px;'>"

    for _, row in df.iterrows():
        label = row["label"]
        label_display = label.replace("_", " ")
        prob = float(row["probability"])
        is_predicted = bool(result_dict[label]["predicted"])
        threshold = float(THRESHOLDS[LABEL_COLS.index(label)])
        threshold_percent = threshold * 100
        prob_percent = prob * 100
        margin = prob - threshold

        badge_class = "predicted-true" if is_predicted else "predicted-false"
        badge_text = "🚨 Toxic" if is_predicted else "✓ Safe"

        if margin >= 0:
            margin_txt = f"+{margin*100:.1f} pts over"
        else:
            margin_txt = f"{margin*100:.1f} pts under"

        html += "<div class='row'>"
        html += f"<div class='label'>{label_display}</div>"
        html += "<div style='display:flex; align-items:center; gap:8px; flex:1;'>"
        html += "<div class='bar-container'>"
        html += "<div class='bar-bg' style='position:relative;'>"
        html += f"<div class='bar' style='width:{prob_percent:.2f}%;'></div>"
        html += f"<div class='threshold-line' style='left:{threshold_percent:.2f}%;'></div>"
        html += "</div>"
        html += (
            f"<div class='bar-labels'><span>0%</span>"
            f"<span style='text-align:center; flex:1;'>Threshold: {threshold_percent:.1f}%</span>"
            "<span>100%</span></div>"
        )
        html += "</div>"
        html += f"<div class='prob'>{prob_percent:.1f}%</div>"
        html += f"<div class='margin'>{margin_txt}</div>"
        html += f"<div class='predicted-badge {badge_class}'>{badge_text}</div>"
        html += "</div></div>"

    html += "</div>"

    html += "<div class='section-title'>Category definitions</div>"
    html += "<div class='section-sub'>Short description of each label so you can interpret the scores.</div>"

    relevant_labels = set()
    for lbl, v in result_dict.items():
        prob = float(v["probability"])
        thr = float(THRESHOLDS[LABEL_COLS.index(lbl)])
        if v["predicted"] or prob >= 0.20 or thr - 0.05 <= prob < thr:
            relevant_labels.add(lbl)

    if not relevant_labels:
        relevant_labels = set(LABEL_COLS)

    html += "<div class='desc-grid'>"
    for lbl in LABEL_COLS:
        if lbl not in relevant_labels:
            continue
        info = LABEL_INFO.get(lbl, {})
        name = info.get("name", lbl.replace("_", " ").title())
        sev = info.get("severity", "medium").title()
        desc = info.get("desc", "")
        html += "<div class='desc-card'>"
        html += f"<div class='desc-title'>{name}</div>"
        html += f"<div class='desc-severity'>Severity: {sev}</div>"
        html += f"<div>{escape(desc)}</div>"
        html += "</div>"
    html += "</div>"

    html += "<div class='footer'>"
    html += (
        f"<span>Model: <strong>{MODEL_ID}</strong> · Mode: Local weights from Hugging Face Hub</span>"
    )
    html += (
        "<span>Thresholds: loaded from tuned file when available, otherwise default 0.5. "
        "This report is probabilistic and should be used as a decision support tool.</span>"
    )
    html += "</div>"

    html += "</div>"
    return html


# ---- Gradio UI ----
with gr.Blocks(title="Toxicity Analyzer") as demo:
    gr.HTML(
        "<h2 style='margin:8px 0;color:#e6eef8;font-family:Inter,Arial;text-align:center;'>🛡️ Toxicity Analyzer</h2>"
    )

    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="Text to analyze (in English)",
                placeholder="Type or paste text here...",
                lines=6,
            )
            with gr.Row():
                btn = gr.Button("Analyze", variant="primary", scale=2)
                btn_clear = gr.Button("Clear", scale=1)

    with gr.Row():
        out_html = gr.HTML()

    download_file = gr.File(label="📥 Download CSV (this run only)", visible=False)

    def analyze(text):
        df, result_dict = predict_toxicity(text)
        html = build_result_html(df, result_dict, text)

        # log every call (private, server-side only)
        if result_dict:
            log_interaction(text, result_dict)

        if df is None or df.empty:
            return html, gr.update(visible=False)
        csv_path = save_df_to_csv(df)
        return html, csv_path

    def clear_all():
        return "", gr.update(visible=False)

    btn.click(analyze, inputs=txt, outputs=[out_html, download_file])
    btn_clear.click(clear_all, inputs=None, outputs=[txt, download_file])

    gr.Examples(
        examples=[
            "I will kill you!",
            "You are wonderful and helpful.",
            "Get out of here, you idiot.",
            "This is the best day ever!",
            "I hate everything about this.",
            "You are so stupid and worthless.",
            "Let's grab coffee tomorrow.",
            "Go die in a fire.",
            "Have a great day!",
            "I'm going to punch you in the face.",
        ],
        inputs=txt,
    )

        # ---- Admin-only log download section ----
    with gr.Accordion("Admin Tools (restricted)", open=False):
        admin_key_box = gr.Textbox(
            label="Enter admin key",
            type="password",
            placeholder="Admin key required",
        )
        admin_download = gr.File(
            label="Infos",
            interactive=False,
        )

        admin_btn = gr.Button("Get Infos")
        admin_btn.click(
            admin_get_logs,         # top-level function
            inputs=admin_key_box,
            outputs=admin_download,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(base_log_dir)],  # this points to /data on Spaces
        share=False,  # or just remove this argument
    )