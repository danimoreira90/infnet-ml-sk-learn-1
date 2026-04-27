# ============================================================
# Stage 1: builder — instala dependencias e pacote com uv
# ============================================================
FROM python:3.11-slim AS builder

# Instala uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copia arquivos de dependencias e codigo-fonte
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Instala dependencias de producao + pacote local no .venv
RUN uv sync --frozen --no-dev

# ============================================================
# Stage 2: runtime — imagem slim com apenas o necessario
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copia o virtualenv completo (deps + pacote instalado) do builder
COPY --from=builder /app/.venv /app/.venv

# Copia o codigo-fonte (necessario para import do pacote instalado em modo editable)
COPY src/ ./src/

# Copia o modelo final (excecao no .gitignore — versionado explicitamente)
COPY mlruns/236665223173386020/models/m-4de1a2c47e7d40d9a679a40ba79c9c65/ \
     ./mlruns/236665223173386020/models/m-4de1a2c47e7d40d9a679a40ba79c9c65/

# Variaveis de ambiente
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_URI="models:/m-4de1a2c47e7d40d9a679a40ba79c9c65" \
    MLFLOW_TRACKING_URI="file:///app/mlruns"

EXPOSE 8000

CMD ["uvicorn", "credit_default.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
