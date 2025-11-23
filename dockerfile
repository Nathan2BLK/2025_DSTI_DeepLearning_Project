FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer Python libs
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code et les poids du modèle
COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
