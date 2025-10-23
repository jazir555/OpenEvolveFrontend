FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY *.json ./
COPY *.md ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Initialize database
RUN python -c "from sovereign_persistence import SovereignDatabase; SovereignDatabase().init_database()"

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from sovereign_reliability import get_health_monitor; m = get_health_monitor(); exit(0 if m.run_health_checks()['overall_healthy'] else 1)"

# Run the application
CMD ["streamlit", "run", "api_server.py", "--server.port=8501", "--server.address=0.0.0.0"]
