FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Variables de entorno para evitar warnings
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY preclip.py .
COPY clip_task.py .
COPY compare.py .

# Crear directorio para dataset y resultados
RUN mkdir -p /app/dataset /app/results

# El dataset debe montarse desde el host
VOLUME ["/app/dataset", "/app/results"]

# Comando por defecto: mostrar ayuda
CMD ["echo", "Uso: docker run -v ./dataset:/app/dataset -v ./results:/app/results practica2-clip python3 <script.py>"]