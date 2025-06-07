FROM python:3.11-slim-bookworm

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements-2.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-2.txt

# Copy app source code
COPY ./AI /app/AI
WORKDIR /app/AI

# RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')"

# Entry point
CMD ["python3", "main.py"]
