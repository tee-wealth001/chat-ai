FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for building or running torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 7860 required by Hugging Face Spaces
EXPOSE 7860

# Run FastAPI app with Uvicorn on port 7860 (required by Spaces)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
