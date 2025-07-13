# Stage 1: Use PyTorch base image
FROM pytorch-base:latest as web-stage

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