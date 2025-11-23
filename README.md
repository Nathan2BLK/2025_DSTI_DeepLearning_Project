# 🧠 2025_DSTI_DeepLearning_Project  
**Toxic Comment Classification with Transformers (Jigsaw Dataset)**  

## 📘 Project Overview  

This project implements a multi-label toxic comment classification system using a fine-tuned BERT model and the the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) dataset. The application uses Gradio to provide an interactive web interface where users can input text and get both probability scores for each toxic category (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and the predicted toxic labels.

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

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Python](https://www.python.org/)


## Project Structure

- `app.py` : Gradio apps for inference.
- `data/processed/`: Preprocessed datasets (cleaned data)
- `test/`: Test predictions and thresholds.
- `toxicity_final.ipynb`: Notebook for exploration and model training
- `dockerfile`: Docker setup for containerized inference.
- `requirements.txt`: Python dependencies.

> Note: Docker is configured only for inference; training should be done locally or on a separate environment.

## Requirements

Minimal Python dependencies:

torch==2.9.1
transformers==4.57.1
numpy==2.2.6
pandas==2.3.3
datasets==4.4.1
gradio
gradio_client

## Running with Docker

1. Build the Docker image:

```bash
docker build -t toxic-gradio .
```
2. Run the docker container

```bash
docker run --rm -p 7860:7860 toxic-gradio
```

The application will be available http://localhost:7860
Gradio will also generate a public link gradio.live that you can use for 1 week

## Running locally without docker

```bash
pip install -r requirements.txt
python app.py
```
The application will be available on http://localhost:7860


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
