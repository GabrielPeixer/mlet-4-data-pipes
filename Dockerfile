# Imagem base enxuta com Python
FROM python:3.11-slim

# Evita arquivos .pyc e garante logs sem buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instala dependências primeiro (melhora cache de build)
COPY requirements.txt .
# Instala a versão CPU-only do PyTorch (evita baixar pacotes CUDA de vários GB)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copia o restante do código e artefatos do modelo
COPY . .

# A maioria das plataformas (Cloud Run, Render, Railway) injeta a porta via $PORT
ENV PORT=8000
EXPOSE 8000

# Sobe a API FastAPI com uvicorn, respeitando a porta da plataforma
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
