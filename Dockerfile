# Stage 1: Build PyTorch base image
# Using Debian 12 (Bookworm) for better stability
FROM python:3.10-slim-bookworm as pytorch-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

RUN pip install --no-cache-dir -U pip

# Install PyTorch and scientific computing packages
# Updated to latest stable versions for better EfficientNet support
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.1 \
    tensorboard==2.15.1

# Create directories for volumes
RUN mkdir -p /workspace/app /workspace/models /workspace/data /workspace/logs

# Stage 2: Build the application image
FROM pytorch-base as web-stage

# Install Gradio with compatible versions (confirmed working)
RUN pip install --no-cache-dir \
    gradio==4.44.0 \
    fastapi==0.104.1 \
    pydantic==2.5.0 \
    uvicorn==0.24.0 \
    starlette==0.27.0

# Install remaining dependencies with updated versions
RUN pip install --no-cache-dir \
    tqdm==4.66.1 \
    seaborn==0.13.2 \
    matplotlib==3.10.3 \
    onnx==1.16.0 \
    onnxruntime==1.17.1 \
    nvidia-tensorrt

# Copy application files
COPY classifier_app.py /workspace/

# Expose Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "/workspace/classifier_app.py"]