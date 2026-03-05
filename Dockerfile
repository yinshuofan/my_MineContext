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
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . .

# Install the project itself (no dependencies needed as they are already installed)
# This installs the project into the virtual environment created in the previous step
RUN uv sync --frozen

# Ensure subsequent commands use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Install playwright dependencies (uncomment if needed)
# Since playwright is in pyproject.toml, it is installed by uv sync.
# You only need to install the browsers:
# RUN playwright install --with-deps chromium

# Create directories for logs and data
RUN mkdir -p logs persist screenshots

# Expose port
EXPOSE 1733

# Entrypoint
ENTRYPOINT ["python"]
CMD ["-m", "opencontext.cli", "start", "--config", "config/config-docker.yaml"]
