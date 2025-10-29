# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system libraries needed by TensorFlow/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Command to run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 