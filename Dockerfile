FROM python:3.11-slim

# HuggingFace Spaces requires port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY environment/ ./environment/
COPY app.py .
COPY openenv.yaml .
COPY inference.py .

# Create __init__ files for package discovery
RUN touch environment/__init__.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
