# Dockerfile for Customer Churn Prediction & LTV Engine
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Run FastAPI
CMD ["uvicorn", "src/main:app", "--host", "0.0.0.0", "--port", "8000"]