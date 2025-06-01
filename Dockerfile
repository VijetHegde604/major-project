FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements-2.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-2.txt


COPY ./AI /app/AI
WORKDIR /app/AI


RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')"


CMD ["python", "main.py"]
