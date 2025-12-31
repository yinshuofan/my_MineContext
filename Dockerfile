# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTEXT_PATH=/app \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Set working directory
WORKDIR /app

# Install system dependencies
# RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install playwright dependencies
# RUN pip install playwright && \
#     playwright install --with-deps chromium

# Create directories for logs and data
RUN mkdir -p logs persist screenshots

# Expose port
EXPOSE 1733

# Entrypoint
ENTRYPOINT ["python"]
CMD ["-m", "opencontext.cli", "start", "--config", "config/config.yaml"]
