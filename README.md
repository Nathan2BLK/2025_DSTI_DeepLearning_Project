# 🧠 2025_DSTI_DeepLearning_Project  
**Toxic Comment Classification with Transformers (Jigsaw Dataset)**  

---

## 📘 Project Overview  

This project applies **Deep Learning for NLP** to detect multiple categories of **toxic online comments** using the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) dataset.  

It fine-tunes a **Transformer-based model (DistilBERT/BERT)** to perform **multi-label text classification** across six categories of toxicity:  
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

The notebook implements a complete pipeline:
- Data preprocessing & exploratory analysis  
- Transformer fine-tuning (DistilBERT vs BERT benchmark)  
- Hyperparameter optimization and early stopping  
- Evaluation (macro/weighted F1, ROC-AUC)  
- Inference & Gradio demo  
- Kaggle submission generation  
- Ethical considerations and reproducibility  

---

## 🧩 Dataset  

**Source:** Kaggle – *Jigsaw Toxic Comment Classification Challenge*  
**Files used:**
- `train.csv` – training comments with labels  
- `test.csv` – test comments (no labels)  
- `test_labels.csv` – hidden test labels (subset released for evaluation)  
- `sample_submission.csv` – Kaggle submission format  

| Label | Description |
|:--|:--|
| `toxic` | General offensive language |
| `severe_toxic` | High-intensity toxicity |
| `obscene` | Use of profanity or sexual terms |
| `threat` | Violent intent |
| `insult` | Personal attacks |
| `identity_hate` | Hate speech targeting identity groups |

Class imbalance is handled using **inverse-frequency label weighting** in the loss function.

---

## ⚙️ Methodology  

### 1. Model Selection  
A benchmark phase compares **DistilBERT** and **BERT-base-uncased** on a subset of the dataset to balance accuracy and computational efficiency.  
The model with the highest `macro-F1` is selected for optimization.

### 2. Fine-Tuning  
- **Loss:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)  
- **Optimizer:** AdamW (learning rate `2e-5`, weight decay `0.01`)  
- **Scheduler:** Linear warmup  
- **Training:** 2–3 epochs with early stopping  
- **Batch size:** 16 (GPU) / 8 (CPU)  
- **Preprocessing:** Lowercasing, tokenization, truncation (max 192 tokens)

### 3. Optimization  
Hyperparameters tuned: learning rate, batch size, number of epochs, decision threshold.  
Final threshold optimized on the validation set for maximum macro-F1.

---

## 📊 Evaluation  

### Metrics
| Metric | Description |
|:--|:--|
| **Macro F1** | Mean F1 across labels (primary metric) |
| **Weighted F1** | Weighted by label prevalence |
| **ROC-AUC** | Probabilistic separation between classes |
| **Per-label F1** | Detailed performance by toxicity type |

### Results Summary (example)
| Model | Macro-F1 | Weighted-F1 | ROC-AUC (macro) |
|:--|:--:|:--:|:--:|
| DistilBERT | 0.86 | 0.91 | 0.94 |
| BERT-base | 0.87 | 0.92 | 0.95 |

*(Values may vary slightly depending on seed and subset size.)*

Validation and Kaggle test-subset scores are reported automatically in the notebook.  

---

## 💬 Inference & Demo  

A **Gradio** web interface provides real-time toxicity detection:  

```python
gr.Interface(
    fn=classify_comment,
    inputs=["textbox", "slider"],
    outputs=["label", "json"],
    title="Jigsaw Toxic Comment Classifier",
    description="DistilBERT/BERT multi-label classifier with sigmoid outputs."
).launch()
```

You can paste any comment and see per-label probabilities with adjustable threshold.

---

## 📈 Kaggle Submission  

The notebook automatically generates a valid submission file:  

```bash
submission.csv
```

containing probabilities for each of the six labels.  
Upload it to the Kaggle competition page to receive your public leaderboard score.

---

## ⚖️ Ethical Considerations  

- **Bias & Fairness:**  
  The model reflects societal biases in the data (e.g., gendered or cultural slurs).  
  Mitigation includes threshold tuning and class weighting.  

- **Transparency:**  
  All code, preprocessing steps, and metrics are fully documented for reproducibility.  

- **Privacy & Responsible Use:**  
  This model is for academic research on moderation; it should **not** be used for real-world moderation decisions without human oversight.  

- **Sustainability:**  
  DistilBERT is used to minimize compute and energy cost while maintaining high accuracy.

---

## 🧮 Reproducibility  

**Environment:**  
- Python ≥ 3.10  
- PyTorch ≥ 2.0  
- Transformers ≥ 4.38  
- Datasets, Evaluate, TorchMetrics, Gradio  

**Training Configuration:**  
- Random seed = 42  
- Early stopping (patience = 2)  
- Model artifacts saved to `/outputs_jigsaw`  

To reproduce:
```bash
pip install -r requirements.txt
python toxic_comment_notebook.ipynb
```

---

## 📦 Project Structure  

```
project_jigsaw/
│
├── train.csv
├── test.csv
├── test_labels.csv
├── sample_submission.csv
├── toxic_comment_notebook.ipynb
├── outputs_jigsaw/
│   └── distilbert-base-uncased_BEST/
├── submission.csv
└── README.md
```

---

## 🏁 Key Takeaways  

- Transformers excel at detecting nuanced toxic language.  
- Multi-label setup allows simultaneous recognition of multiple offense types.  
- Careful threshold tuning and class weighting improve minority-class recall.  
- Lightweight DistilBERT achieves competitive performance with lower energy usage.  
- The project demonstrates full reproducibility and ethical awareness — aligned with DSTI 2025 Deep Learning project standards.  

---

**Author:** *Nathan De Blecker*  
**Institution:** Data ScienceTech Institute (DSTI) — MSc Data Science & Engineering  
**Course:** Deep Learning with Python (2025)  
