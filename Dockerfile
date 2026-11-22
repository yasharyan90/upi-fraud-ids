FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . && python -m upi_ids.cli train --out results/models/hybrid_detector.pkl
EXPOSE 8000
CMD ["python", "-m", "upi_ids.cli", "serve", "--model", "results/models/hybrid_detector.pkl", "--port", "8000"]
