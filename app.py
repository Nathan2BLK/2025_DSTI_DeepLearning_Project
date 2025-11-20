import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Charger le modèle et le tokenizer
model_path = "./bert-base-uncased_epoch4"  # Met ici le chemin de ton modèle fine-tuné
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Liste des labels dans l'ordre utilisé à l'entraînement
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_toxic(text):
    # Prédire sur du texte brut
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        # On fixe un seuil typique à 0.5 pour la classification multi-label
        pred_labels = [labels[i] for i, p in enumerate(probs) if p > 0.5]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}, ", ".join(pred_labels) if pred_labels else "No toxic label detected"

interface = gr.Interface(
    fn=predict_toxic,
    inputs=gr.Textbox(lines=3, placeholder="Enter comment here...", label="Comment"),
    outputs=[
        gr.Label(label="Label Probabilities (Sigmoid)"),
        gr.Textbox(label="Predicted Toxic Labels")
    ],
    title="Toxic Comment Classification (BERT-base-uncased)"
)

interface.launch()