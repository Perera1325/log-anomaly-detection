FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000 8501
# bootstrap is idempotent: generates sample data + trains only if models are missing
CMD ["sh", "-c", "python bootstrap.py && uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
