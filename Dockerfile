FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

WORKDIR /app

# Install runtime dependencies
COPY pyproject.toml setup.py README.md requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and install the package in editable mode
COPY easy_mlops /app/easy_mlops
COPY bin /app/bin

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["easy-mlops"]
CMD ["--help"]
