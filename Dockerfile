FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# downloads CUAD (~18 MB) and trains the linear model if no model is mounted at ./artifacts
RUN test -f artifacts/clause_classifier.joblib || python scripts/train.py --fast

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
