# RESE Phase II: Isomorphic Mapping Adapter
# Dockerfile for containerized deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY ../schemas/ /app/glue/schemas/
COPY ../lib/ /app/glue/lib/

# Create non-root user
RUN useradd -m -u 1000 rese && \
    chown -R rese:rese /app
USER rese

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.phase2_adapter import Phase2Adapter; adapter = Phase2Adapter(); adapter.get_health()" || exit 1

# Default command
CMD ["python", "-u", "src/phase2_adapter.py", "--health"]

# Expose port (if running as API server)
EXPOSE 8000
