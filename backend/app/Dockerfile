# Usar una versión de Python compatible con mediapipe
FROM python:3.10-slim

# Evitar buffering de stdout/stderr
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los requirements
COPY app/requirements.txt .

# Instalar dependencias del sistema necesarias para OpenCV/mediapipe
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar todo el backend al contenedor
COPY app/ .

# Comando para iniciar tu app en Render
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
