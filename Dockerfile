# Use official Python image
FROM python:3.10-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent python from buffering stdout
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create required folders
RUN mkdir -p data/processed outputs outputs/models outputs/metrics outputs/figures outputs/recommendations

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]