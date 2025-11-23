# 2025_DSTI_DeepLearning_Project
Deep Learning NLP project using the Jigsaw Toxic Comment dataset. Fine-tunes a Transformer to detect multiple toxicity types (toxic, obscene, insult, etc.). Includes preprocessing, training, evaluation, and a lightweight demo for real-time text classification and reproducibility.

## Prerequisites
- Docker Desktop (https://www.docker.com/products/docker-desktop/)
- Python (https://www.python.org/)
This project implements a multi-label toxic comment classification system using a fine-tuned BERT model. The application uses Gradio to provide an interactive web interface where users can input text and get both probability scores for each toxic category (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and the predicted toxic labels above a 0.5 threshold.

## Project Structure

- `app.py`: Gradio app for inference.
- `train.py`: Script converted from notebook for local training (optional).
- `results_bert/`: Directory containing the fine-tuned BERT model.
- `Dockerfile`: Docker configuration for containerized deployment.
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
docker run --rm -p 7860:7860 toxic-gradio
```
## Running locally without docker
```bash
pip install -r requirements.txt
python app.py
```