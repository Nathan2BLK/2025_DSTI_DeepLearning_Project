# 🧠 2025_DSTI_DeepLearning_Project  
**Toxic Comment Classification with Transformers (Jigsaw Dataset)**  

## 📘 Project Overview  

This project implements a multi-label toxic comment classification system using a fine-tuned BERT model and the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) dataset. The application uses Gradio to provide an interactive web interface where users can input text and get both probability scores for each toxic category (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and the predicted toxic labels.

The notebook implements a complete pipeline:
- Data preprocessing & exploratory analysis  
- Transformer fine-tuning (DistilBERT vs BERT benchmark)  
- Hyperparameter optimization and early stopping  
- Evaluation (macro/weighted F1, ROC-AUC)  
---

## 🧩 Dataset  

**Source:** Kaggle –  [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) 

**Files used:**
- `train.csv` – training comments with labels  
- `test.csv` – test comments (no labels)  
- `test_labels.csv` – hidden test labels (subset released for evaluation)

| Label | Description |
|:--|:--|
| `toxic` | General offensive language |
| `severe_toxic` | High-intensity toxicity |
| `obscene` | Use of profanity or sexual terms |
| `threat` | Violent intent |
| `insult` | Personal attacks |
| `identity_hate` | Hate speech targeting identity groups |

We face class imbalance within the dataset and resolve it by choosing the right model.

## Project Structure

- `app.py` : Gradio apps for inference
- `data/processed/`: Preprocessed datasets (cleaned data)
- `data`: Row datasets
- `test/`: Test probabilities on predictions and thresholds associated to test
- `test.py`: File to test the pretrained model on a sentence of your choice
- `toxicity_final.ipynb`: Notebook for exploration and model training
- `dockerfile`: Docker setup for containerized inference.
- `requirements.txt`: Python dependencies.

> Note: Docker is configured only for inference; training should be done locally or on a separate environment.

## ⚙️ Methodology  

### 1. Model Selection  
A benchmark phase compares a baseline mode **TF-IDF + Logistic Regression** to  two adavanced transformer models **DistilBERT** and **BERT-base-uncased** on a subset of the dataset to balance accuracy and computational efficiency.  
The model with the highest `macro-F1` is selected for optimization.

### 2. Fine-Tuning  
- **Loss:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)  
- **Optimizer:** AdamW (learning rate `2e-5`, weight decay `0.01`)  
- **Scheduler:** Linear warmup  
- **Training:** 4-10 epochs with early stopping  
- **Batch size:** 16 / 32
- **Preprocessing:** Lowercasing, tokenization, truncation (max 128 tokens)

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
| Model | Macro-F1 |
|:--|:--|
| Baseline (TF-IDF + LogReg) | 0.54 |
| DistilBERT | 0.63 |
| BERT-base | 0.67 | 

*(Values may vary slightly depending on seed and subset size.)*

Validation and Kaggle test-subset scores are reported automatically in the notebook.  

---

## 💬 Inference & Demo  

A **Gradio** web interface provides real-time toxicity detection, look at app.py

Here is the link to access to the web interface: [link](https://huggingface.co/spaces/NathanDB/toxic-bert-dsti)
You can click on one of the suggested comments or write a new one to see the detailed toxicity analysis.

---

## 🧮 Reproducibility  

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Python](https://www.python.org/)

## Requirements

Minimal Python dependencies:
-torch==2.9.1
-transformers==4.57.1
-numpy==2.2.6
-pandas==2.3.3
-datasets==4.4.1
-gradio
-gradio_client 

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
Gradio will also generate a public link gradio.live that you can use for 1 week
