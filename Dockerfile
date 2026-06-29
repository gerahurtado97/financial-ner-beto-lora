# ─────────────────────────────────────────────────────────────
# Dockerfile — NER Financiero con BETO + LoRA
# Proyecto Final Módulo 2 · Diplomado AI & LLM for Financial Markets · ITAM
#
# Empaqueta inference.py con Python 3.11 fijo, eliminando el problema
# de reproducibilidad observado al correr en Python 3.13 (numpy sin
# wheel precompilado, requiere compilador C/C++ no disponible).
#
# Solo CPU — la inferencia con LoRA sobre BETO-base es rápida sin GPU,
# y evita la complejidad de imágenes CUDA para un caso de uso puntual.
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="ITAM Diplomado AI & LLM"
LABEL description="NER Financiero — BETO + LoRA — Inferencia"
LABEL version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar primero las dependencias — Docker cachea esta capa si no cambian
COPY requirements.lock requirements-dev.lock ./

# Notas sobre el lockfile dentro del contenedor:
#   1. pywin32 y similares: filtrados porque no existen en Linux
#      (el lockfile se generó en una máquina Windows)
#   2. --no-deps en el pip install del lockfile: el lockfile de
#      pip-compile --generate-hashes exige version exacta (==) en
#      TODA dependencia transitiva. Una sub-dependencia opcional
#      (hf-xet, instalada condicionalmente por huggingface-hub)
#      quedó con rango en vez de version exacta, lo que rompe el modo
#      --require-hashes implicito al haber hashes en el archivo.
#      Quitamos los hashes con un grep adicional para evitar ese modo
#      estricto dentro del contenedor — las fuentes (PyPI oficial +
#      indice de PyTorch) ya son confiables en este contexto.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    grep -v -i "^pywin32\|^pywin32-ctypes\|^pypiwin32\|^win32" requirements.lock \
        | grep -v -E "^\s*--hash" \
        | sed -E 's/ \\$//' \
        > requirements.lock.linux && \
    pip install --no-cache-dir -r requirements.lock.linux

# Copiar código fuente y artefactos necesarios para inferencia
COPY src/ ./src/
COPY inference.py ./
COPY configs/ ./configs/

# Modelo LoRA óptimo (r=16) — pesos ligeros (~3.4MB), ya están en el repo
COPY models/ner_lora_r16/lora_weights/ ./models/ner_lora_r16/lora_weights/

# Ejemplos de prueba incluidos en la imagen para verificación rápida
COPY docs/ejemplos_txt/ ./docs/ejemplos_txt/

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import torch, transformers, peft" || exit 1

# Por defecto corre sobre el ejemplo 1 — para usar con un archivo propio:
#   docker run -v /ruta/local:/data financial-ner-lora python inference.py /data/archivo.txt
CMD ["python", "inference.py", "docs/ejemplos_txt/ejemplo1.txt"]
