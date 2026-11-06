FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y modelos
COPY app/ ./app/
COPY models/ ./models/

# Exponer el puerto estándar que usa Hugging Face Spaces (7860) o el puerto 8000
EXPOSE 7860

# Iniciar Uvicorn en el puerto 7860 (requerido por Hugging Face)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
