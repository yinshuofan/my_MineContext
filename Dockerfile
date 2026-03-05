# Use uv base image which includes python 3.13
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set environment variables
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTEXT_PATH=/app \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    # Also set UV_INDEX_URL if using uv
    UV_INDEX_URL=${PIP_INDEX_URL} \
    UV_COMPILE_BYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (uncomment if needed later)
# RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy dependency definition files first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into a virtual environment
# --frozen ensures we use the lockfile
# --no-install-project installs only dependencies first, so we can cache this layer
# --mount=type=cache reuses downloaded packages across builds even when uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy project files
COPY . .

# Install the project itself (no dependencies needed as they are already installed)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Ensure subsequent commands use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 1733

# Entrypoint
ENTRYPOINT ["python"]
CMD ["-m", "opencontext.cli", "start", "--config", "config/config-docker.yaml"]
