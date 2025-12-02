# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies required for llvmlite
# llvmlite requires llvm-config which is usually in llvm-dev or similar packages
# Install system dependencies
# We rely on manylinux wheels for llvmlite, so we don't need to install LLVM manually
# unless we are building from source.
# RUN apt-get update && apt-get install -y build-essential llvm && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY examples/ examples/
COPY tests/ tests/
COPY run_tests.py .

# Install dependencies
# We use pip directly here for simplicity in the container
RUN pip install --no-cache-dir llvmlite typer rich pytest syrupy

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Entry point
ENTRYPOINT ["python", "-m", "forgec.main"]
CMD ["--help"]
