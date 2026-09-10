FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model during the Docker build
RUN python src/10_train_model.py

EXPOSE 10000

CMD ["sh", "-c", "python -m uvicorn src.10_api:app --host 0.0.0.0 --port ${PORT:-10000}"]