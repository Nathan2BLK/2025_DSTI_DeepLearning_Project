# 2025_DSTI_DeepLearning_Project
This project implements a multi-label toxic comment classification system using a fine-tuned BERT model. The application uses Gradio to provide an interactive web interface where users can input text and get both probability scores for each toxic category (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and the predicted toxic labels.

## Prerequisites
- [Docker Desktop] (https://www.docker.com/products/docker-desktop/)
- [Python] (https://www.python.org/)


## Project Structure

- `app.py`: Gradio apps for inference.
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

The application will be available [here](http://localhost:7860). Gradio will also generate a public link gradio.live that you can use for 1 week

## Running locally without docker

```bash
pip install -r requirements.txt
python app.py
```
The application will be available [here](http://localhost:7860).