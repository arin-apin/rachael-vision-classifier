# 🔬 PyTorch Image Classifier - Rachael AI Vision Platform

A professional PyTorch-based image classification platform with web interface, supporting multiple architectures, data augmentation, model quantization, and format conversion.

## 🚀 Features

- **Multiple Model Architectures**: ResNet, MobileNet, EfficientNet families
- **Transfer Learning**: Leverages pre-trained models for faster training
- **Data Augmentation**: Configurable augmentation pipeline
- **Model Optimization**: INT8 and FP16 quantization support
- **Format Conversion**: Export to ONNX, TorchScript, and TensorRT
- **Web Interface**: User-friendly Gradio interface
- **Multi-language Support**: English and Spanish (configurable)
- **GPU Acceleration**: CUDA support for training and inference
- **Real-time Training Monitoring**: Live progress tracking

## 📋 Requirements

- Docker and Docker Compose
- NVIDIA GPU (optional but recommended)
- NVIDIA Docker runtime (for GPU support)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/arin/pytorch-classifier.git
cd pytorch-classifier
```

2. Build and run with Docker Compose:
```bash
docker compose up --build -d
```

3. Access the interface at `http://localhost:7860`

## 📁 Project Structure

```
pytorch-classifier/
├── classifier_app.py       # Main application
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── README.md             # This file
├── models/               # Trained models directory
├── data/                 # Dataset directory
└── logs/                 # Training logs
```

## 🎯 Usage

### 1. Dataset Preparation

Prepare your dataset as a ZIP file with the following structure:
```
dataset.zip
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

### 2. Training

1. Upload your dataset ZIP file
2. Select a base model (ResNet, MobileNet, or EfficientNet)
3. Configure data augmentation options
4. Set training parameters (epochs, learning rate)
5. Click "Start Training"

### 3. Model Optimization

- **Quantization**: Convert models to INT8 or FP16 for deployment
- **Format Conversion**: Export to ONNX, TorchScript, or TensorRT

### 4. Inference

1. Select a trained model
2. Upload an image
3. Get detailed classification results with confidence scores

## 🔧 Configuration

### Language Setting

Edit `classifier_app.py` line 182:
```python
CURRENT_LANGUAGE = 'en'  # Options: 'en' (English) or 'es' (Spanish)
```

### GPU Configuration

The system automatically detects and uses CUDA if available. No configuration needed.

## 🤖 Supported Models

### ResNet Family
- ResNet-18 (11.7M params) - Fast, good for prototyping
- ResNet-34 (21.8M params) - Balanced performance
- ResNet-50 (25.6M params) - Industry standard

### MobileNet Family
- MobileNet-V2 (3.5M params) - Classic mobile architecture
- MobileNet-V3 Small (2.5M params) - Ultra-lightweight
- MobileNet-V3 Large (5.5M params) - Best mobile accuracy

### EfficientNet Family
- EfficientNet-B0 (5.3M params) - Baseline efficiency
- EfficientNet-B2 (9.1M params) - Better accuracy
- EfficientNet-B3 (12M params) - High accuracy

## 📊 Performance Tips

1. **For Mobile/Edge Deployment**: Use MobileNet or quantized models
2. **For Best Accuracy**: Use EfficientNet-B3 or ResNet-50
3. **For Fast Training**: Use ResNet-18 or MobileNet-V3 Small
4. **Data Augmentation**: Enable for better generalization

## 🐛 Troubleshooting

### Container won't start
- Check Docker logs: `docker logs pytorch-classifier`
- Ensure ports are not in use: `lsof -i :7860`

### CUDA out of memory
- Reduce batch size in training
- Use a smaller model
- Close other GPU applications

### Model not appearing after training
- Check `models/` directory permissions
- Ensure training completed successfully
- Refresh the model list in the interface

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏢 About

Developed by Arin using Rachael AI Vision technology.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Contact: [support@arin.com]

---

**Version**: 1.0.0  
**PyTorch**: 2.1.0  
**CUDA**: 11.8 (optional)