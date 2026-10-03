FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py app.py review_core.py combined_review.py security_reference.py .
COPY OWASP_Cheatsheets_Book.pdf .

EXPOSE 8000 7860

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
