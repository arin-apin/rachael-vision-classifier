import gradio as gr
import zipfile
import os
import tempfile
import shutil
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import threading
import time
import locale

# Language detection and translation system
def detect_language():
    """Detect user language from system locale, fallback to Spanish"""
    try:
        # Try to get system locale
        system_locale = locale.getdefaultlocale()[0]
        if system_locale and system_locale.startswith('en'):
            return 'en'
        else:
            return 'es'  # Default to Spanish
    except:
        return 'es'  # Fallback to Spanish

# Translation dictionaries
TRANSLATIONS = {
    'es': {
        # Main interface
        'title': '🔬 Clasificador de Imágenes con PyTorch',
        'upload_zip': '📁 Subir Dataset (ZIP)',
        'upload_zip_info': 'Sube un ZIP con carpetas para cada clase',
        'model_selection': '🤖 Selección de Modelo',
        'data_augmentation': '🎨 Data Augmentation',
        'use_augmentation': 'Usar Data Augmentation',
        'rotation': 'Rotación',
        'max_degrees': 'Grados máximos',
        'zoom': 'Zoom',
        'zoom_range': 'Rango de zoom (min)',
        'brightness_contrast': 'Brillo/Contraste',
        'brightness_range': 'Factor de brillo',
        'contrast_range': 'Factor de contraste',
        'horizontal_flip': 'Volteo Horizontal',
        'vertical_flip': 'Volteo Vertical',
        'training_params': '⚙️ Parámetros de Entrenamiento',
        'epochs': 'Épocas',
        'learning_rate': 'Tasa de Aprendizaje',
        'train_button': '🚀 Entrenar Modelo',
        'training_progress': '📈 Progreso de Entrenamiento',
        'training_tab': '🎯 Entrenamiento',
        'metrics_tab': '📊 Métricas',
        'quantization_tab': '⚡ Cuantización',
        'conversion_tab': '🔄 Conversión',
        'prediction_tab': '🔮 Predicción',
        'select_model': 'Seleccionar modelo entrenado:',
        'refresh': 'Actualizar',
        'load_metrics': '📊 Cargar Métricas',
        'quantize_model': '⚡ Cuantizar Modelo',
        'convert_model': '🔄 Convertir Modelo',
        'upload_image': '📸 Subir Imagen para Predicción',
        'predict_button': '🔮 Predecir',
        'prediction_result': '🎯 Resultado de Predicción',
        'model_info': '📋 Información del Modelo',
        # Results and messages
        'analysis_title': '# 🔬 **Análisis de Clasificación**',
        'main_prediction': '## 🎯 **Predicción Principal**',
        'detailed_results': '## 📊 **Resultados Detallados**',
        'all_classes': '*Todas las clases ordenadas por confianza:*',
        'confidence_analysis': '## 🧠 **Análisis de Confianza**',
        'technical_details': '## ⚙️ **Detalles Técnicos**',
        'confidence_very_high': '✅ **Confianza Muy Alta** - El modelo está muy seguro de esta predicción',
        'confidence_high': '✅ **Confianza Alta** - El modelo está confiado en esta predicción',
        'confidence_moderate': '⚠️ **Confianza Moderada** - El modelo tiene cierta incertidumbre',
        'confidence_low': '❌ **Confianza Baja** - El modelo está inseguro, considera revisar la calidad de la imagen',
        'model_label': 'Modelo',
        'classes_label': 'Clases',
        'device_label': 'Dispositivo',
        'image_size_label': 'Tamaño imagen',
        'confidence_label': 'confianza',
        'total_label': 'total',
        'normalized_label': 'normalizada',
        'no_model_trained': '❌ No hay modelo entrenado',
        'model_loaded': '✅ Modelo cargado',
        'with_classes': 'con',
        'error_loading_model': '❌ Error cargando modelo',
        'dataset_valid': '✅ Dataset válido',
        'classes_text': 'clases',
        'images_text': 'imágenes',
        'need_folders': '❌ Necesitas 2+ carpetas con imágenes',
        'few_images': '❌ Muy pocas imágenes',
        'minimum': 'Mínimo',
        'error_text': '❌ Error',
        # Interface elements
        'dataset_tab': 'Dataset',
        'upload_validate_title': 'Subir y validar dataset',
        'upload_validate_desc': 'Sube un archivo ZIP con imágenes organizadas en carpetas por clase.',
        'validate_button': 'Validar Dataset',
        'validation_result': 'Resultado de validación',
        'config_training_title': 'Configurar y ejecutar entrenamiento',
        'base_model_label': 'Modelo base',
        'base_model_info': 'Selecciona el modelo base para transfer learning',
        'data_aug_title': 'Data Augmentation',
        'training_params_title': 'Parámetros de Entrenamiento',
        'start_training': 'Iniciar Entrenamiento',
        'training_plots': 'Gráficas de Entrenamiento',
        'trained_model_label': 'Modelo entrenado',
        'model_to_quantize': 'Modelo a cuantizar',
        'model_to_convert': 'Modelo a convertir',
        'image_to_classify': 'Imagen a clasificar',
        'convert_models_title': 'Convertir modelos a diferentes formatos'
    },
    'en': {
        # Main interface
        'title': '🔬 PyTorch Image Classifier',
        'upload_zip': '📁 Upload Dataset (ZIP)',
        'upload_zip_info': 'Upload a ZIP with folders for each class',
        'model_selection': '🤖 Model Selection',
        'data_augmentation': '🎨 Data Augmentation',
        'use_augmentation': 'Use Data Augmentation',
        'rotation': 'Rotation',
        'max_degrees': 'Max degrees',
        'zoom': 'Zoom',
        'zoom_range': 'Zoom range (min)',
        'brightness_contrast': 'Brightness/Contrast',
        'brightness_range': 'Brightness factor',
        'contrast_range': 'Contrast factor',
        'horizontal_flip': 'Horizontal Flip',
        'vertical_flip': 'Vertical Flip',
        'training_params': '⚙️ Training Parameters',
        'epochs': 'Epochs',
        'learning_rate': 'Tasa de Aprendizaje',
        'train_button': '🚀 Train Model',
        'training_progress': '📈 Training Progress',
        'training_tab': '🎯 Training',
        'metrics_tab': '📊 Metrics',
        'quantization_tab': '⚡ Quantization',
        'conversion_tab': '🔄 Conversion',
        'prediction_tab': '🔮 Prediction',
        'select_model': 'Select trained model:',
        'refresh': 'Refresh',
        'load_metrics': '📊 Load Metrics',
        'quantize_model': '⚡ Quantize Model',
        'convert_model': '🔄 Convert Model',
        'upload_image': '📸 Upload Image for Prediction',
        'predict_button': '🔮 Predict',
        'prediction_result': '🎯 Prediction Result',
        'model_info': '📋 Model Information',
        # Results and messages
        'analysis_title': '# 🔬 **Classification Analysis**',
        'main_prediction': '## 🎯 **Main Prediction**',
        'detailed_results': '## 📊 **Detailed Results**',
        'all_classes': '*All classes ordered by confidence:*',
        'confidence_analysis': '## 🧠 **Confidence Analysis**',
        'technical_details': '## ⚙️ **Technical Details**',
        'confidence_very_high': '✅ **Very High Confidence** - The model is very sure about this prediction',
        'confidence_high': '✅ **High Confidence** - The model is confident in this prediction',
        'confidence_moderate': '⚠️ **Moderate Confidence** - The model has some uncertainty',
        'confidence_low': '❌ **Low Confidence** - The model is unsure, consider reviewing image quality',
        'model_label': 'Model',
        'classes_label': 'Classes',
        'device_label': 'Device',
        'image_size_label': 'Image size',
        'confidence_label': 'confidence',
        'total_label': 'total',
        'normalized_label': 'normalized',
        'no_model_trained': '❌ No trained model',
        'model_loaded': '✅ Model loaded',
        'with_classes': 'with',
        'error_loading_model': '❌ Error loading model',
        'dataset_valid': '✅ Valid dataset',
        'classes_text': 'classes',
        'images_text': 'images',
        'need_folders': '❌ Need 2+ folders with images',
        'few_images': '❌ Too few images',
        'minimum': 'Minimum',
        'error_text': '❌ Error',
        # Interface elements
        'dataset_tab': 'Dataset',
        'upload_validate_title': 'Upload and validate dataset',
        'upload_validate_desc': 'Upload a ZIP file with images organized in folders by class.',
        'validate_button': 'Validate Dataset',
        'validation_result': 'Validation Result',
        'config_training_title': 'Configure and run training',
        'base_model_label': 'Base Model',
        'base_model_info': 'Select the base model for transfer learning',
        'data_aug_title': 'Data Augmentation',
        'training_params_title': 'Training Parameters',
        'start_training': 'Start Training',
        'training_plots': 'Training Plots',
        'trained_model_label': 'Trained Model',
        'model_to_quantize': 'Model to Quantize',
        'model_to_convert': 'Model to Convert',
        'image_to_classify': 'Image to Classify',
        'convert_models_title': 'Convert models to different formats',
        # Additional interface elements
        'subtitle_professional_en': '🚀 **Professional PyTorch Image Classification Suite**',
        'subtitle_advanced_en': '*Advanced Computer Vision Solutions*',
        'subtitle_professional_es': '🚀 **Suite Profesional de Clasificación de Imágenes con PyTorch**',
        'subtitle_advanced_es': '*Soluciones Avanzadas de Visión por Computadora*',
        'config_training_desc': 'Configurar y ejecutar entrenamiento',
        'training_parameters_title': '⚙️ Parámetros de Entrenamiento',
        'epochs_label': '📊 Épocas',
        'data_augmentation_title': '🎨 Data Augmentation',
        'use_data_augmentation': 'Usar Data Augmentation',
        'rotation_label': 'Rotación',
        'max_degrees_label': 'Grados máximos',
        'zoom_label': 'Zoom',
        'zoom_range_label': 'Rango de zoom (min)',
        'brightness_contrast_label': 'Brillo/Contraste',
        'brightness_range_label': 'Rango de brillo (min)',
        'horizontal_flip_label': 'Volteo Horizontal',
        'vertical_flip_label': 'Volteo Vertical',
        'start_training_button': '🚀 Iniciar Entrenamiento',
        'training_result_label': '📋 Resultado del entrenamiento',
        'training_plots_label': '📊 Gráficas de Entrenamiento',
        'model_info_label': '📊 Información del modelo',
        'metrics_tab_title': '📈 Métricas',
        'metrics_training_desc': 'Métricas y gráficas del entrenamiento',
        'trained_model_dropdown': '📂 Modelo entrenado',
        'model_metrics_info': 'Selecciona un modelo para ver sus métricas',
        'refresh_button': 'Actualizar',
        'load_metrics_button': '📊 Cargar Métricas',
        'metrics_summary_label': '📋 Resumen de Métricas',
        'select_trained_model': 'Selecciona un modelo entrenado',
        'quantization_tab_title': '⚡ Cuantización',
        'quantization_title': 'Cuantizar modelos entrenados para optimización',
        'quantization_desc': 'Reduce el tamaño del modelo y acelera la inferencia para deployment en dispositivos edge.',
        'model_to_quantize_dropdown': '📂 Modelo a cuantizar',
        'model_quantize_info': 'Selecciona un modelo entrenado para cuantizar',
        'quantization_type_label': '🔧 Tipo de cuantización',
        'quantization_type_info': 'INT8 reduce más el tamaño, FP16 mantiene mejor precisión',
        'int8_option': 'INT8 - Máxima compresión (dynamic quantization)',
        'fp16_option': 'FP16 - Balance entre tamaño y precisión',
        'quantize_model_button': '⚡ Cuantizar Modelo',
        'quantization_result_label': '📋 Resultado de cuantización',
        'conversion_tab_title': '🔄 Conversión',
        'conversion_title': 'Convertir modelos a diferentes formatos',
        'conversion_desc': 'Exporta modelos entrenados a ONNX, TorchScript u otros formatos para deployment.',
        'model_to_convert_dropdown': '📂 Modelo a convertir',
        'model_convert_info': 'Selecciona un modelo entrenado para convertir',
        'target_format_label': '📋 Formato de destino',
        'target_format_info': 'ONNX: universal, TorchScript: PyTorch, TensorRT: NVIDIA GPU',
        'onnx_option': 'ONNX - Estándar multiplataforma',
        'torchscript_option': 'TorchScript - Optimizado para PyTorch',
        'tensorrt_option': 'TensorRT - Máximo rendimiento NVIDIA GPU',
        'convert_model_button': '🔄 Convertir Modelo',
        'conversion_result_label': '📋 Resultado de conversión',
        'prediction_tab_title': '🔍 Predicción',
        'prediction_title': 'Clasificación de imágenes',
        'prediction_desc': 'Selecciona un modelo entrenado y sube una imagen para análisis detallado.',
        'prediction_model_dropdown': '📂 Modelo entrenado',
        'prediction_model_info': 'Selecciona un modelo para hacer predicciones',
        'image_to_classify_label': '🖼️ Imagen a clasificar',
        'classify_button': '🎯 Clasificar',
        'analysis_result_label': '📝 Resultado del análisis',
        'select_model_first': '❌ Selecciona un modelo entrenado primero',
        'upload_image_first': '❌ Sube una imagen para clasificar',
        'rachael_platform_title': '🔬 **Plataforma de Visión Rachael AI**',
        'professional_solutions': 'Soluciones Profesionales de Visión por Computadora',
        'gpu_status': '🔧 **Estado GPU:**',
        'gpu_enabled': '✅ GPU NVIDIA Habilitada',
        'cpu_mode': '❌ Modo CPU',
        'features_title': '🚀 **Características:**',
        'features_list': 'Aprendizaje Profundo • Transfer Learning • Optimización de Modelos • Aceleración TensorRT',
        'more_info': '🌐 **Más Información:**',
        'version_info': '**Versión:** PyTorch 2.1.0',
        'copyright_text': '*© 2024 Rachael AI - Tecnología Avanzada de Visión por Computadora*'
    },
    'en': {
        # Main interface
        'title': '🔬 PyTorch Image Classifier',
        'upload_zip': '📁 Upload Dataset (ZIP)',
        'upload_zip_info': 'Upload a ZIP with folders for each class',
        'model_selection': '🤖 Model Selection',
        'data_augmentation': '🎨 Data Augmentation',
        'use_augmentation': 'Use Data Augmentation',
        'rotation': 'Rotation',
        'max_degrees': 'Max degrees',
        'zoom': 'Zoom',
        'zoom_range': 'Zoom range (min)',
        'brightness_contrast': 'Brightness/Contrast',
        'brightness_range': 'Brightness factor',
        'contrast_range': 'Contrast factor',
        'horizontal_flip': 'Horizontal Flip',
        'vertical_flip': 'Vertical Flip',
        'training_params': '⚙️ Training Parameters',
        'epochs': 'Epochs',
        'learning_rate': 'Learning Rate',
        'train_button': '🚀 Train Model',
        'training_progress': '📈 Training Progress',
        'training_tab': '🎯 Training',
        'metrics_tab': '📊 Metrics',
        'quantization_tab': '⚡ Quantization',
        'conversion_tab': '🔄 Conversion',
        'prediction_tab': '🔮 Prediction',
        'select_model': 'Select trained model:',
        'refresh': 'Refresh',
        'load_metrics': '📊 Load Metrics',
        'quantize_model': '⚡ Quantize Model',
        'convert_model': '🔄 Convert Model',
        'upload_image': '📸 Upload Image for Prediction',
        'predict_button': '🔮 Predict',
        'prediction_result': '🎯 Prediction Result',
        'model_info': '📋 Model Information',
        # Results and messages
        'analysis_title': '# 🔬 **Classification Analysis**',
        'main_prediction': '## 🎯 **Main Prediction**',
        'detailed_results': '## 📊 **Detailed Results**',
        'all_classes': '*All classes ordered by confidence:*',
        'confidence_analysis': '## 🧠 **Confidence Analysis**',
        'technical_details': '## ⚙️ **Technical Details**',
        'confidence_very_high': '✅ **Very High Confidence** - The model is very sure about this prediction',
        'confidence_high': '✅ **High Confidence** - The model is confident in this prediction',
        'confidence_moderate': '⚠️ **Moderate Confidence** - The model has some uncertainty',
        'confidence_low': '❌ **Low Confidence** - The model is unsure, consider reviewing image quality',
        'model_label': 'Model',
        'classes_label': 'Classes',
        'device_label': 'Device',
        'image_size_label': 'Image size',
        'confidence_label': 'confidence',
        'total_label': 'total',
        'normalized_label': 'normalized',
        'no_model_trained': '❌ No trained model',
        'model_loaded': '✅ Model loaded',
        'with_classes': 'with',
        'error_loading_model': '❌ Error loading model',
        'dataset_valid': '✅ Valid dataset',
        'classes_text': 'classes',
        'images_text': 'images',
        'need_folders': '❌ Need 2+ folders with images',
        'few_images': '❌ Too few images',
        'minimum': 'Minimum',
        'error_text': '❌ Error',
        # Interface elements
        'dataset_tab': 'Dataset',
        'upload_validate_title': 'Upload and validate dataset',
        'upload_validate_desc': 'Upload a ZIP file with images organized in folders by class.',
        'validate_button': 'Validate Dataset',
        'validation_result': 'Validation Result',
        'config_training_title': 'Configure and run training',
        'base_model_label': 'Base Model',
        'base_model_info': 'Select the base model for transfer learning',
        'data_aug_title': 'Data Augmentation',
        'training_params_title': 'Training Parameters',
        'start_training': 'Start Training',
        'training_plots': 'Training Plots',
        'trained_model_label': 'Trained Model',
        'model_to_quantize': 'Model to Quantize',
        'model_to_convert': 'Model to Convert',
        'image_to_classify': 'Image to Classify',
        'convert_models_title': 'Convert models to different formats',
        # Additional interface elements
        'subtitle_professional_en': '🚀 **Professional PyTorch Image Classification Suite**',
        'subtitle_advanced_en': '*Advanced Computer Vision Solutions*',
        'subtitle_professional_es': '🚀 **Professional PyTorch Image Classification Suite**',
        'subtitle_advanced_es': '*Advanced Computer Vision Solutions*',
        'config_training_desc': 'Configure and run training',
        'training_parameters_title': '⚙️ Training Parameters',
        'epochs_label': '📊 Epochs',
        'data_augmentation_title': '🎨 Data Augmentation',
        'use_data_augmentation': 'Use Data Augmentation',
        'rotation_label': 'Rotation',
        'max_degrees_label': 'Max degrees',
        'zoom_label': 'Zoom',
        'zoom_range_label': 'Zoom range (min)',
        'brightness_contrast_label': 'Brightness/Contrast',
        'brightness_range_label': 'Brightness range (min)',
        'horizontal_flip_label': 'Horizontal Flip',
        'vertical_flip_label': 'Vertical Flip',
        'start_training_button': '🚀 Start Training',
        'training_result_label': '📋 Training Result',
        'training_plots_label': '📊 Training Plots',
        'model_info_label': '📊 Model Information',
        'metrics_tab_title': '📈 Metrics',
        'metrics_training_desc': 'Training metrics and plots',
        'trained_model_dropdown': '📂 Trained Model',
        'model_metrics_info': 'Select a model to view its metrics',
        'refresh_button': 'Refresh',
        'load_metrics_button': '📊 Load Metrics',
        'metrics_summary_label': '📋 Metrics Summary',
        'select_trained_model': 'Select a trained model',
        'quantization_tab_title': '⚡ Quantization',
        'quantization_title': 'Quantize trained models for optimization',
        'quantization_desc': 'Reduce model size and accelerate inference for edge device deployment.',
        'model_to_quantize_dropdown': '📂 Model to Quantize',
        'model_quantize_info': 'Select a trained model to quantize',
        'quantization_type_label': '🔧 Quantization Type',
        'quantization_type_info': 'INT8 reduces size more, FP16 maintains better precision',
        'int8_option': 'INT8 - Maximum compression (dynamic quantization)',
        'fp16_option': 'FP16 - Balance between size and precision',
        'quantize_model_button': '⚡ Quantize Model',
        'quantization_result_label': '📋 Quantization Result',
        'conversion_tab_title': '🔄 Conversion',
        'conversion_title': 'Convert models to different formats',
        'conversion_desc': 'Export trained models to ONNX, TorchScript or other formats for deployment.',
        'model_to_convert_dropdown': '📂 Model to Convert',
        'model_convert_info': 'Select a trained model to convert',
        'target_format_label': '📋 Target Format',
        'target_format_info': 'ONNX: universal, TorchScript: PyTorch, TensorRT: NVIDIA GPU',
        'onnx_option': 'ONNX - Cross-platform standard',
        'torchscript_option': 'TorchScript - Optimized for PyTorch',
        'tensorrt_option': 'TensorRT - Maximum NVIDIA GPU performance',
        'convert_model_button': '🔄 Convert Model',
        'conversion_result_label': '📋 Conversion Result',
        'prediction_tab_title': '🔍 Prediction',
        'prediction_title': 'Image classification',
        'prediction_desc': 'Select a trained model and upload an image for detailed analysis.',
        'prediction_model_dropdown': '📂 Trained Model',
        'prediction_model_info': 'Select a model to make predictions',
        'image_to_classify_label': '🖼️ Image to Classify',
        'classify_button': '🎯 Classify',
        'analysis_result_label': '📝 Analysis Result',
        'select_model_first': '❌ Select a trained model first',
        'upload_image_first': '❌ Upload an image to classify',
        'rachael_platform_title': '🔬 **Rachael AI Vision Platform**',
        'professional_solutions': 'Professional Computer Vision Solutions',
        'gpu_status': '🔧 **GPU Status:**',
        'gpu_enabled': '✅ NVIDIA GPU Enabled',
        'cpu_mode': '❌ CPU Mode',
        'features_title': '🚀 **Features:**',
        'features_list': 'Deep Learning • Transfer Learning • Model Optimization • TensorRT Acceleration',
        'more_info': '🌐 **More Information:**',
        'version_info': '**Version:** PyTorch 2.1.0',
        'copyright_text': '*© 2024 Rachael AI - Advanced Computer Vision Technology*'
    }
}

# Global language setting - will be overridden below

def t(key, lang=None):
    """Translate function"""
    if lang is None:
        lang = CURRENT_LANGUAGE
    return TRANSLATIONS.get(lang, TRANSLATIONS['es']).get(key, key)

def get_model_descriptions(lang=None):
    """Get model descriptions in specified language"""
    if lang is None:
        lang = CURRENT_LANGUAGE
    
    if lang == 'en':
        return {
            "resnet18": {
                "description": "Lightweight and fast, ideal for prototyping",
                "params": "11.7M parameters",
                "speed": "Very fast (1ms inference)",
                "accuracy": "Good (70-85% typical)",
                "edge_quantized": "~11MB INT8, <1ms on CPU",
                "use_case": "Mobile, IoT, real-time"
            },
            "resnet34": {
                "description": "Balance between speed and accuracy",
                "params": "21.8M parameters", 
                "speed": "Fast (2ms inference)",
                "accuracy": "Very good (75-88% typical)",
                "edge_quantized": "~21MB INT8, 1-2ms on CPU",
                "use_case": "Web applications, edge computing"
            },
            "resnet50": {
                "description": "Industry standard, excellent accuracy",
                "params": "25.6M parameters",
                "speed": "Moderate (3-4ms inference)", 
                "accuracy": "Excellent (80-92% typical)",
                "edge_quantized": "~25MB INT8, 2-3ms on CPU",
                "use_case": "Production, high accuracy"
            },
            "mobilenet_v2": {
                "description": "Classic mobile with inverted residuals",
                "params": "3.5M parameters",
                "speed": "Very fast (1ms inference)",
                "accuracy": "Good (70-83% typical)",
                "edge_quantized": "~3.5MB INT8, <1ms on CPU", 
                "use_case": "Classic mobile, IoT"
            },
            "mobilenet_v3_small": {
                "description": "Ultra-lightweight with neural architecture search",
                "params": "2.5M parameters",
                "speed": "Ultra fast (<1ms inference)",
                "accuracy": "Good (68-80% typical)",
                "edge_quantized": "~2.5MB INT8, <0.5ms on CPU",
                "use_case": "Ultra mobile, extreme edge"
            },
            "mobilenet_v3_large": {
                "description": "Efficient mobile with best accuracy",
                "params": "5.5M parameters",
                "speed": "Fast (1-2ms inference)",
                "accuracy": "Very good (75-86% typical)",
                "edge_quantized": "~5.5MB INT8, <1ms on CPU",
                "use_case": "Mobile premium, good balance"
            },
            "efficientnet_b0": {
                "description": "Compound scaling baseline",
                "params": "5.3M parameters",
                "speed": "Moderate (2-3ms inference)",
                "accuracy": "Very good (77-88% typical)",
                "edge_quantized": "~5MB INT8, 1-2ms on CPU",
                "use_case": "Balanced efficiency"
            },
            "efficientnet_b2": {
                "description": "Scaled efficiency with better accuracy",
                "params": "9.1M parameters", 
                "speed": "Moderate (3-4ms inference)",
                "accuracy": "Excellent (80-91% typical)",
                "edge_quantized": "~9MB INT8, 2-3ms on CPU",
                "use_case": "Production, efficiency focused"
            },
            "efficientnet_b3": {
                "description": "High accuracy with compound scaling",
                "params": "12M parameters",
                "speed": "Slower (4-6ms inference)",
                "accuracy": "Exceptional (82-93% typical)",
                "edge_quantized": "~12MB INT8, 3-4ms on CPU",
                "use_case": "High accuracy, research"
            }
        }
    else:  # Spanish
        return {
            "resnet18": {
                "description": "Modelo ligero y rápido, ideal para prototipado",
                "params": "11.7M parámetros",
                "speed": "Muy rápido (1ms inference)",
                "accuracy": "Buena (70-85% típico)",
                "edge_quantized": "~11MB INT8, <1ms en CPU",
                "use_case": "Móviles, IoT, tiempo real"
            },
            "resnet34": {
                "description": "Balance entre velocidad y precisión",
                "params": "21.8M parámetros",
                "speed": "Rápido (2ms inference)",
                "accuracy": "Muy buena (75-88% típico)",
                "edge_quantized": "~21MB INT8, 1-2ms en CPU",
                "use_case": "Aplicaciones web, edge computing"
            },
            "resnet50": {
                "description": "Estándar de la industria, excelente precisión",
                "params": "25.6M parámetros",
                "speed": "Moderado (3-4ms inference)",
                "accuracy": "Excelente (80-92% típico)",
                "edge_quantized": "~25MB INT8, 2-3ms en CPU",
                "use_case": "Producción, alta precisión"
            },
            "mobilenet_v2": {
                "description": "Clásico móvil con inverted residuals",
                "params": "3.5M parámetros",
                "speed": "Muy rápido (1ms inference)",
                "accuracy": "Buena (70-83% típico)",
                "edge_quantized": "~3.5MB INT8, <1ms en CPU",
                "use_case": "Móviles clásico, IoT"
            },
            "mobilenet_v3_small": {
                "description": "Ultra-ligero con neural architecture search",
                "params": "2.5M parámetros",
                "speed": "Ultra rápido (<1ms inference)",
                "accuracy": "Buena (68-80% típico)",
                "edge_quantized": "~2.5MB INT8, <0.5ms en CPU",
                "use_case": "Ultra móvil, edge extremo"
            },
            "mobilenet_v3_large": {
                "description": "Móvil eficiente con mejor precisión",
                "params": "5.5M parámetros",
                "speed": "Rápido (1-2ms inference)",
                "accuracy": "Muy buena (75-86% típico)",
                "edge_quantized": "~5.5MB INT8, <1ms en CPU",
                "use_case": "Móvil premium, buen balance"
            },
            "efficientnet_b0": {
                "description": "Baseline de escalado compuesto",
                "params": "5.3M parámetros",
                "speed": "Moderado (2-3ms inference)",
                "accuracy": "Muy buena (77-88% típico)",
                "edge_quantized": "~5MB INT8, 1-2ms en CPU",
                "use_case": "Eficiencia balanceada"
            },
            "efficientnet_b2": {
                "description": "Eficiencia escalada con mejor precisión",
                "params": "9.1M parámetros",
                "speed": "Moderado (3-4ms inference)",
                "accuracy": "Excelente (80-91% típico)",
                "edge_quantized": "~9MB INT8, 2-3ms en CPU",
                "use_case": "Producción, enfoque en eficiencia"
            },
            "efficientnet_b3": {
                "description": "Alta precisión con escalado compuesto",
                "params": "12M parámetros",
                "speed": "Lento (4-6ms inference)",
                "accuracy": "Excepcional (82-93% típico)",
                "edge_quantized": "~12MB INT8, 3-4ms en CPU",
                "use_case": "Alta precisión, investigación"
            }
        }

# Set language based on environment - for Docker deployment, default to English
CURRENT_LANGUAGE = 'en'  # Force English for testing, can be changed by editing this line

# Simple trainer class (embedded for simplicity)
class SimpleTrainer:
    def __init__(self):
        self.model = None
        self.classes = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create timestamped directories for each training session
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"/workspace/models/{self.timestamp}"
        self.model_path = f"{self.session_dir}/model.pth"
        self.plots_path = f"{self.session_dir}/training_plots.png"
        self.history_path = f"{self.session_dir}/training_history.json"
        self.config_path = f"{self.session_dir}/config.json"
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs("/workspace/data", exist_ok=True)
        self.training_history = None
        self.training_progress = {
            'current_epoch': 0,
            'total_epochs': 0,
            'current_batch': 0,
            'total_batches': 0,
            'current_loss': 0.0,
            'current_acc': 0.0,
            'phase': 'idle'  # idle, training, validation
        }
    
    def extract_and_validate_zip(self, zip_file):
        """Extract ZIP and validate structure"""
        if zip_file is None:
            return None, f"❌ {t('upload_zip')}"
        
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            classes = {}
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            
            with zipfile.ZipFile(zip_file.name, 'r') as z:
                z.extractall(temp_dir)
                
                for path in z.namelist():
                    if '/' in path and not path.endswith('/'):
                        folder = path.split('/')[0]
                        filename = path.split('/')[-1]
                        ext = os.path.splitext(filename)[1].lower()
                        
                        if ext in valid_extensions:
                            classes[folder] = classes.get(folder, 0) + 1
            
            if len(classes) < 2:
                shutil.rmtree(temp_dir)
                return None, t('need_folders')
            
            total = sum(classes.values())
            if total < 10:
                shutil.rmtree(temp_dir)
                return None, f"{t('few_images')} ({total}). {t('minimum')}: 10"
            
            # Store classes
            self.classes = sorted(list(classes.keys()))
            
            # Store dataset information for config
            self.dataset_name = os.path.basename(zip_file.name).replace('.zip', '') if hasattr(zip_file, 'name') else 'Custom Dataset'
            self.total_images = total
            self.classes_distribution = classes
            self.dataset_source = f"ZIP file: {os.path.basename(zip_file.name)}" if hasattr(zip_file, 'name') else 'User uploaded ZIP'
            
            result = f"{t('dataset_valid')}: {len(classes)} {t('classes_text')}, {total} {t('images_text')}\n"
            for k, v in classes.items():
                result += f"• {k}: {v} {t('images_text')}\n"
            
            return temp_dir, result
            
        except Exception as e:
            return None, f"{t('error_text')}: {str(e)}"
    
    def get_available_models(self):
        """Get available models with detailed information"""
        # Get translated descriptions
        model_descs = get_model_descriptions()
        
        # Build model info with translations
        models_info = {}
        
        # Add each model with size info
        for model_key in ["resnet18", "resnet34", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b2", "efficientnet_b3"]:
            if model_key in model_descs:
                model_names = {
                    "resnet18": "ResNet-18",
                    "resnet34": "ResNet-34", 
                    "resnet50": "ResNet-50",
                    "mobilenet_v2": "MobileNet-V2",
                    "mobilenet_v3_small": "MobileNet-V3 Small",
                    "mobilenet_v3_large": "MobileNet-V3 Large",
                    "efficientnet_b0": "EfficientNet-B0",
                    "efficientnet_b2": "EfficientNet-B2",
                    "efficientnet_b3": "EfficientNet-B3"
                }
                
                model_sizes = {
                    "resnet18": "~45MB",
                    "resnet34": "~83MB", 
                    "resnet50": "~98MB",
                    "mobilenet_v2": "~14MB",
                    "mobilenet_v3_small": "~10MB",
                    "mobilenet_v3_large": "~21MB",
                    "efficientnet_b0": "~20MB",
                    "efficientnet_b2": "~31MB",
                    "efficientnet_b3": "~43MB"
                }
                
                models_info[model_key] = {
                    "name": model_names[model_key],
                    **model_descs[model_key],
                    "size": model_sizes[model_key]
                }
        
        return models_info
    
    def get_model_info(self, model_name):
        """Get model information for display"""
        models = self.get_available_models()
        
        if model_name not in models:
            if CURRENT_LANGUAGE == 'en':
                return "Select a model to view detailed information"
            else:
                return "Selecciona un modelo para ver información detallada"
        
        model = models[model_name]
        
        if CURRENT_LANGUAGE == 'en':
            info = f"""📊 **Model Information**
        
🤖 **{model['name']}**
{model['description']}

📋 **Specifications:**
• **Size:** {model['size']}
• **Parameters:** {model['params']}
• **Speed:** {model['speed']}
• **Expected accuracy:** {model['accuracy']}

⚡ **INT8 Quantization (Edge/Mobile):**
• **Optimized size:** {model['edge_quantized']}

🎯 **Ideal use case:** {model['use_case']}"""
        else:
            info = f"""📊 **Información del Modelo**
        
🤖 **{model['name']}**
{model['description']}

📋 **Especificaciones:**
• **Tamaño:** {model['size']}
• **Parámetros:** {model['params']}
• **Velocidad:** {model['speed']}
• **Precisión esperada:** {model['accuracy']}

⚡ **Cuantización INT8 (Edge/Mobile):**
• **Tamaño optimizado:** {model['edge_quantized']}

🎯 **Caso de uso ideal:** {model['use_case']}"""
        
        return info
    
    def create_model(self, num_classes, model_name="resnet18"):
        """Create model based on selection"""
        import torch.hub
        # Clear cache to avoid hash issues
        torch.hub.set_dir('/tmp/torch_cache')
        
        try:
            # ResNet Family
            if model_name == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif model_name == "resnet34":
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif model_name == "resnet50":
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # MobileNet Family
            elif model_name == "mobilenet_v2":
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            elif model_name == "mobilenet_v3_small":
                model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            elif model_name == "mobilenet_v3_large":
                model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            
            # EfficientNet V1 Family
            elif model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b1":
                model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b2":
                model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b3":
                model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b4":
                model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b5":
                model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b6":
                model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_b7":
                model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            
            # EfficientNet V2 Family
            elif model_name == "efficientnet_v2_s":
                model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_v2_m":
                model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            elif model_name == "efficientnet_v2_l":
                model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            
            else:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Falling back to random initialization...")
            # Fallback to random initialization if pretrained fails
            model_constructors = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
                "mobilenet_v2": models.mobilenet_v2,
                "mobilenet_v3_small": models.mobilenet_v3_small,
                "mobilenet_v3_large": models.mobilenet_v3_large,
                "efficientnet_b0": models.efficientnet_b0,
                "efficientnet_b1": models.efficientnet_b1,
                "efficientnet_b2": models.efficientnet_b2,
                "efficientnet_b3": models.efficientnet_b3,
                "efficientnet_b4": models.efficientnet_b4,
                "efficientnet_b5": models.efficientnet_b5,
                "efficientnet_b6": models.efficientnet_b6,
                "efficientnet_b7": models.efficientnet_b7,
                "efficientnet_v2_s": models.efficientnet_v2_s,
                "efficientnet_v2_m": models.efficientnet_v2_m,
                "efficientnet_v2_l": models.efficientnet_v2_l,
            }
            constructor = model_constructors.get(model_name, models.resnet18)
            model = constructor(weights=None)
        
        # Freeze early layers for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        
        # Get the correct input features for each model family
        if "mobilenet" in model_name:
            # MobileNet family uses 'classifier' attribute
            if model_name == "mobilenet_v2":
                # MobileNet V2 has different structure: classifier[1] is the main linear layer
                num_features = model.classifier[1].in_features
            else:
                # MobileNet V3 (small/large): get from first linear layer
                if hasattr(model.classifier, '0') and hasattr(model.classifier[0], 'in_features'):
                    num_features = model.classifier[0].in_features
                else:
                    num_features = model.classifier[-1].in_features
            
            # Replace entire classifier for all MobileNets
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
        elif "efficientnet" in model_name:
            # EfficientNet family uses 'classifier' attribute
            if hasattr(model.classifier, '1') and hasattr(model.classifier[1], 'in_features'):
                # Most EfficientNet models have: Dropout -> Linear
                num_features = model.classifier[1].in_features
            else:
                # Fallback to last layer
                num_features = model.classifier[-1].in_features
            
            # Simple classifier for EfficientNet (they're already very efficient)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, num_classes)
            )
            
        else:  # ResNet models
            # ResNet family uses 'fc' attribute
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        return model.to(self.device)
    
    def train_model(self, data_dir, epochs=5, lr=0.001, progress_callback=None, augmentation_config=None):
        """Training function with progress tracking"""
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from tqdm import tqdm
        
        # Generate new timestamp for this training session
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"/workspace/models/{self.timestamp}"
        self.model_path = f"{self.session_dir}/model.pth"
        self.plots_path = f"{self.session_dir}/training_plots.png"
        self.history_path = f"{self.session_dir}/training_history.json"
        self.config_path = f"{self.session_dir}/config.json"
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Dataset class
        class SimpleDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.data_dir = data_dir
                self.transform = transform
                self.images = []
                self.labels = []
                
                for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
                    class_path = os.path.join(data_dir, class_name)
                    if os.path.isdir(class_path):
                        for img_name in os.listdir(class_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                self.images.append(os.path.join(class_path, img_name))
                                self.labels.append(idx)
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img_path = self.images[idx]
                image = Image.open(img_path).convert('RGB')
                label = self.labels[idx]
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        # Data transforms with augmentation based on user preferences
        transform_list = [transforms.Resize((224, 224))]
        
        # Store augmentation config for dataset info
        self.augmentation_config = augmentation_config if augmentation_config else {}
        
        # Use augmentation config if provided
        if augmentation_config and augmentation_config.get('use_augmentation', True):
            if augmentation_config.get('rotation_enabled', False):
                transform_list.append(transforms.RandomRotation(
                    degrees=augmentation_config.get('rotation_degrees', 15)
                ))
            
            if augmentation_config.get('zoom_enabled', False):
                zoom_range = augmentation_config.get('zoom_range', 0.9)
                transform_list.append(transforms.RandomResizedCrop(
                    224, scale=(zoom_range, 1.0), ratio=(0.8, 1.2)
                ))
            
            if augmentation_config.get('brightness_enabled', False):
                brightness_range = augmentation_config.get('brightness_range', 0.8)
                transform_list.append(transforms.ColorJitter(
                    brightness=(brightness_range, 1/brightness_range),
                    contrast=(brightness_range, 1/brightness_range),
                    saturation=(brightness_range, 1/brightness_range),
                    hue=0.1
                ))
            
            if augmentation_config.get('flip_horizontal', True):
                transform_list.append(transforms.RandomHorizontalFlip())
            
            if augmentation_config.get('flip_vertical', False):
                transform_list.append(transforms.RandomVerticalFlip())
        else:
            # Default: only horizontal flip if no config provided
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Always apply these transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform = transforms.Compose(transform_list)
        
        # Create dataset and dataloader
        dataset = SimpleDataset(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model
        num_classes = len(self.classes)
        model_name = getattr(self, 'selected_model', 'resnet18')
        self.model = self.create_model(num_classes, model_name)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        # Get trainable parameters based on model type
        if hasattr(self.model, 'fc'):
            trainable_params = self.model.fc.parameters()
        elif hasattr(self.model, 'classifier'):
            trainable_params = self.model.classifier.parameters()
        else:
            trainable_params = self.model.parameters()
        optimizer = optim.Adam(trainable_params, lr=lr)
        
        # Training loop
        training_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Split dataset for validation (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Initialize progress tracking
        self.training_progress['total_epochs'] = epochs
        self.training_progress['total_batches'] = len(train_loader)
        
        # For confusion matrix
        all_preds = []
        all_labels = []
        
        for epoch in range(epochs):
            self.training_progress['current_epoch'] = epoch + 1
            self.training_progress['phase'] = 'training'
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                self.training_progress['current_batch'] = batch_idx + 1
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress
                current_acc = 100 * train_correct / train_total
                self.training_progress['current_loss'] = loss.item()
                self.training_progress['current_acc'] = current_acc
                
                if progress_callback and batch_idx % 5 == 0:
                    progress_info = f"Época {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Acc: {current_acc:.1f}%"
                    progress_callback(progress_info)
            
            # Validation phase
            self.training_progress['phase'] = 'validation'
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            # Reset for confusion matrix on last epoch
            if epoch == epochs - 1:
                all_preds = []
                all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Collect for confusion matrix on last epoch
                    if epoch == epochs - 1:
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            # Update history
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_acc'].append(val_acc)
            
            print(f"Época {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save model and history
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'accuracy': val_acc,
            'training_history': training_history,
            'model_name': getattr(self, 'selected_model', 'resnet18')
        }, self.model_path)
        
        # Save history as JSON
        with open(self.history_path, 'w') as f:
            json.dump(training_history, f)
        
        # Save training configuration
        config = {
            'model_name': getattr(self, 'selected_model', 'resnet18'),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'epochs': epochs,
            'learning_rate': lr,
            'device': str(self.device),
            'timestamp': self.timestamp,
            'dataset_info': {
                'name': getattr(self, 'dataset_name', 'Custom Dataset'),
                'total_images': getattr(self, 'total_images', 0),
                'classes_distribution': getattr(self, 'classes_distribution', {}),
                'dataset_source': getattr(self, 'dataset_source', 'User uploaded ZIP'),
                'validation_split': '80/20 train/validation',
                'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
                'preprocessing': {
                    'resize': '224x224',
                    'normalization': 'ImageNet standards',
                    'augmentation_used': getattr(self, 'augmentation_config', {})
                }
            },
            'final_metrics': {
                'train_loss': training_history['train_loss'][-1],
                'train_acc': training_history['train_acc'][-1],
                'val_loss': training_history['val_loss'][-1],
                'val_acc': training_history['val_acc'][-1]
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Generate plots with confusion matrix
        self.generate_training_plots(training_history, all_preds, all_labels)
        
        # Reset progress
        self.training_progress['phase'] = 'idle'
        self.training_history = training_history
        return training_history
    
    def generate_training_plots(self, history, all_preds=None, all_labels=None):
        """Generate training plots with confusion matrix"""
        # Use dark theme for plots
        plt.style.use('dark_background')
        
        # Create figure with more space for confusion matrix
        if all_preds is not None and all_labels is not None:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[0:2, 2])  # Confusion matrix
            ax6 = fig.add_subplot(gs[2, :])    # Progress over time
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = history['epochs']
        
        # Plot 1: Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'g-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'orange', label='Val Acc', linewidth=2)
        ax2.set_title('Precisión durante el Entrenamiento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Combined normalized view
        train_loss_norm = np.array(history['train_loss']) / max(history['train_loss'])
        val_loss_norm = np.array(history['val_loss']) / max(history['val_loss']) if max(history['val_loss']) > 0 else [0]
        train_acc_norm = np.array(history['train_acc']) / 100
        val_acc_norm = np.array(history['val_acc']) / 100
        
        ax3.plot(epochs, train_loss_norm, 'b--', label='Train Loss (norm)', alpha=0.7)
        ax3.plot(epochs, val_loss_norm, 'r--', label='Val Loss (norm)', alpha=0.7)
        ax3.plot(epochs, train_acc_norm, 'g-', label='Train Acc (norm)', linewidth=2)
        ax3.plot(epochs, val_acc_norm, 'orange', label='Val Acc (norm)', linewidth=2)
        ax3.set_title('Vista Normalizada de Métricas', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Valor Normalizado')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning progress
        if len(epochs) > 1:
            train_improvement = np.diff(history['train_acc'])
            val_improvement = np.diff(history['val_acc'])
            
            ax4.bar(epochs[1:], train_improvement, alpha=0.7, label='Mejora Train', color='green')
            ax4.bar(epochs[1:], val_improvement, alpha=0.7, label='Mejora Val', color='orange')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('Mejora por Época', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Época')
            ax4.set_ylabel('Δ Precisión (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Confusion Matrix with dark theme
        if all_preds is not None and all_labels is not None and len(self.classes) > 0:
            cm = confusion_matrix(all_labels, all_preds)
            
            # Use seaborn heatmap with dark-friendly colormap
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                       xticklabels=self.classes, yticklabels=self.classes, ax=ax5)
            ax5.set_title('Matriz de Confusión (Validación)', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Predicción')
            ax5.set_ylabel('Real')
            
            # Plot 6: Detailed progress view
            if len(epochs) > 1:
                # Create detailed epoch view
                batch_points = []
                loss_points = []
                acc_points = []
                
                for i, epoch in enumerate(epochs):
                    # Simulate batch progress within epoch
                    batches_per_epoch = 10  # Simplified view
                    for batch in range(batches_per_epoch):
                        batch_points.append(epoch - 1 + batch/batches_per_epoch)
                        # Interpolate loss and accuracy
                        if i < len(history['train_loss']):
                            loss_points.append(history['train_loss'][i])
                            acc_points.append(history['train_acc'][i])
                
                ax6.plot(batch_points[:len(loss_points)], loss_points, 'b-', alpha=0.7, label='Loss Progress')
                ax6_twin = ax6.twinx()
                ax6_twin.plot(batch_points[:len(acc_points)], acc_points, 'g-', alpha=0.7, label='Accuracy Progress')
                
                ax6.set_xlabel('Época')
                ax6.set_ylabel('Pérdida', color='blue')
                ax6_twin.set_ylabel('Precisión (%)', color='green')
                ax6.set_title('Progreso Detallado del Entrenamiento', fontsize=14, fontweight='bold')
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            return False, t('no_model_trained')
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.classes = checkpoint['classes']
            num_classes = len(self.classes)
            
            # Detect model architecture from checkpoint
            model_name = checkpoint.get('model_name', 'resnet18')
            # Debug prints removed
            
            # Try to detect model architecture from state_dict if model_name is missing or wrong
            if model_name == 'resnet18' and 'model_name' not in checkpoint:  # Default fallback, need to detect
                state_dict_keys = list(checkpoint['model_state_dict'].keys())
                if state_dict_keys:
                    # More sophisticated architecture detection
                    key_string = ' '.join(state_dict_keys)
                    
                    # EfficientNet detection - look for specific patterns
                    if 'features.' in key_string and 'block.' in key_string:
                        if 'features.1.block.0' in key_string:
                            model_name = 'efficientnet_b0'
                        elif any(f'features.{i}.block.' in key_string for i in range(7, 9)):
                            model_name = 'efficientnet_b2'
                        elif any(f'features.{i}.block.' in key_string for i in range(9, 11)):
                            model_name = 'efficientnet_b3'
                        else:
                            model_name = 'efficientnet_b0'  # Default EfficientNet
                    
                    # MobileNet detection
                    elif 'features.' in key_string and ('inverted_residual' in key_string or 'conv_dw' in key_string):
                        model_name = 'mobilenet_v2'
                    
                    # ResNet detection
                    elif 'conv1.' in key_string and 'layer1.' in key_string:
                        if 'layer4.1.' in key_string:
                            model_name = 'resnet34'  # or resnet50
                        else:
                            model_name = 'resnet18'
            
            # Set the detected model architecture
            old_model = getattr(self, 'selected_model', 'resnet18')
            self.selected_model = model_name
            # Model architecture detected and selected
            
            self.model = self.create_model(num_classes, model_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            return True, f"{t('model_loaded')}: {model_name} {t('with_classes')} {num_classes} {t('classes_text')}"
        except Exception as e:
            return False, f"{t('error_loading_model')}: {str(e)}"
    
    def predict(self, image):
        """Make prediction"""
        if self.model is None:
            success, msg = self.load_model()
            if not success:
                return msg
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities sorted by confidence
            all_probs = probabilities[0]  # Get probabilities for all classes
            class_probs = [(self.classes[i], all_probs[i].item()) for i in range(len(self.classes))]
            class_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Build detailed results
            result = f"{t('analysis_title')}\n\n"
            result += f"{t('main_prediction')}\n"
            result += f"**{predicted_class}** - {confidence_score:.2%} {t('confidence_label')}\n\n"
            
            result += f"{t('detailed_results')}\n"
            result += f"{t('all_classes')}\n\n"
            
            for i, (class_name, prob) in enumerate(class_probs):
                if i == 0:
                    # Highlight top prediction
                    result += f"🥇 **{class_name}**: {prob:.2%}\n"
                elif i == 1:
                    result += f"🥈 **{class_name}**: {prob:.2%}\n"
                elif i == 2:
                    result += f"🥉 **{class_name}**: {prob:.2%}\n"
                else:
                    result += f"   {class_name}: {prob:.2%}\n"
            
            # Add confidence interpretation
            result += f"\n{t('confidence_analysis')}\n"
            if confidence_score >= 0.9:
                result += t('confidence_very_high')
            elif confidence_score >= 0.7:
                result += t('confidence_high')
            elif confidence_score >= 0.5:
                result += t('confidence_moderate')
            else:
                result += t('confidence_low')
            
            # Add technical details
            result += f"\n\n{t('technical_details')}\n"
            result += f"- **{t('model_label')}**: {getattr(self, 'selected_model', 'Unknown')}\n"
            result += f"- **{t('classes_label')}**: {len(self.classes)} {t('total_label')}\n"
            result += f"- **{t('device_label')}**: {str(self.device).upper()}\n"
            result += f"- **{t('image_size_label')}**: 224x224 ({t('normalized_label')})\n"
            
            return result
            
        except Exception as e:
            return f"❌ **Error en predicción**: {str(e)}"
    
    def get_model_info(self):
        """Get model information"""
        if not os.path.exists(self.model_path):
            return "❌ No hay modelo entrenado"
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            info = f"""📊 **Información del Modelo**

🔢 **Clases:** {len(checkpoint['classes'])}
📝 **Lista de clases:** {', '.join(checkpoint['classes'])}
🎯 **Precisión:** {checkpoint.get('accuracy', 'N/A'):.2f}%
🖥️ **Dispositivo:** {self.device}
📅 **Archivo:** {os.path.basename(self.model_path)}
"""
            
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']
                info += f"\n📈 **Resumen del Entrenamiento:**\n"
                info += f"• Épocas: {len(history['epochs'])}\n"
                info += f"• Mejor precisión (val): {max(history['val_acc']):.2f}%\n"
                info += f"• Pérdida final (train): {history['train_loss'][-1]:.4f}\n"
                info += f"• Pérdida final (val): {history['val_loss'][-1]:.4f}\n"
            
            return info
            
        except Exception as e:
            return f"❌ Error leyendo información: {str(e)}"
    
    def get_training_plots(self):
        """Load and return training plots"""
        if not os.path.exists(self.plots_path):
            return None
        
        try:
            return Image.open(self.plots_path)
        except Exception as e:
            print(f"Error loading plots: {e}")
            return None
    
    def get_metrics_data(self):
        """Get detailed metrics data"""
        if not os.path.exists(self.history_path):
            return None
        
        try:
            with open(self.history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return None
    
    def get_training_progress(self):
        """Get current training progress"""
        progress = self.training_progress.copy()
        
        # Calculate overall progress
        if progress['total_epochs'] > 0:
            epoch_progress = (progress['current_epoch'] - 1) / progress['total_epochs']
            if progress['total_batches'] > 0:
                batch_progress = progress['current_batch'] / progress['total_batches']
                overall_progress = epoch_progress + (batch_progress / progress['total_epochs'])
            else:
                overall_progress = epoch_progress
        else:
            overall_progress = 0
        
        progress['overall_progress'] = min(overall_progress, 1.0)
        progress['progress_percent'] = progress['overall_progress'] * 100
        
        return progress
    
    def quantize_model(self, model_session_dir, quantization_type="int8"):
        """Quantize trained model to INT8 or FP16"""
        from datetime import datetime
        
        try:
            # Create quantization directory
            quant_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quant_dir = f"/workspace/models/quantized_{quantization_type}_{quant_timestamp}"
            os.makedirs(quant_dir, exist_ok=True)
            
            # Load the trained model
            model_path = f"{model_session_dir}/model.pth"
            if not os.path.exists(model_path):
                return False, "❌ No se encontró modelo entrenado"
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model_name = checkpoint.get('model_name', 'resnet18')
            num_classes = len(checkpoint['classes'])
            
            # Try to detect model architecture from state_dict if model_name is missing or wrong
            if model_name == 'resnet18' and 'model_name' not in checkpoint:  # Default fallback, need to detect
                state_dict_keys = list(checkpoint['model_state_dict'].keys())
                if state_dict_keys:
                    # More sophisticated architecture detection
                    key_string = ' '.join(state_dict_keys)
                    
                    # EfficientNet detection - look for specific patterns
                    if 'features.' in key_string and 'block.' in key_string:
                        if 'features.1.block.0' in key_string:
                            model_name = 'efficientnet_b0'
                        elif any(f'features.{i}.block.' in key_string for i in range(7, 9)):
                            model_name = 'efficientnet_b2'
                        elif any(f'features.{i}.block.' in key_string for i in range(9, 11)):
                            model_name = 'efficientnet_b3'
                        else:
                            model_name = 'efficientnet_b0'  # Default EfficientNet
                    
                    # MobileNet detection
                    elif 'features.' in key_string and ('inverted_residual' in key_string or 'conv_dw' in key_string):
                        model_name = 'mobilenet_v2'
                    
                    # ResNet detection
                    elif 'conv1.' in key_string and 'layer1.' in key_string:
                        if 'layer4.1.' in key_string:
                            model_name = 'resnet34'  # or resnet50
                        else:
                            model_name = 'resnet18'
            
            # Recreate model with detected architecture
            model = self.create_model(num_classes, model_name)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # If loading fails, try to determine the correct architecture
                if 'features.' in str(e) and 'block.' in str(e):
                    # Try different EfficientNet variants
                    for variant in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']:
                        try:
                            model = self.create_model(num_classes, variant)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            model_name = variant
                            break
                        except:
                            continue
                    else:
                        raise RuntimeError(f"Could not load model. Architecture mismatch. Keys suggest EfficientNet but couldn't match variant.")
                elif 'features.' in str(e) and 'classifier.' in str(e):
                    # Try MobileNet variants
                    for variant in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
                        try:
                            model = self.create_model(num_classes, variant)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            model_name = variant
                            break
                        except:
                            continue
                    else:
                        raise RuntimeError(f"Could not load model. Architecture mismatch. Keys suggest MobileNet but couldn't match variant.")
                else:
                    raise e
            
            model.eval()
            
            if quantization_type == "fp16":
                # FP16 quantization
                model = model.half()  # Convert to FP16
                quant_model_path = f"{quant_dir}/quantized_fp16_model.pth"
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classes': checkpoint['classes'],
                    'original_accuracy': checkpoint.get('accuracy', 0),
                    'model_name': model_name,
                    'quantization_info': {
                        'method': 'FP16',
                        'precision': '16-bit floating point',
                        'timestamp': quant_timestamp
                    }
                }, quant_model_path)
                
                method_desc = "FP16 (Half Precision)"
                
            else:  # INT8 quantization
                # Use dynamic quantization for better compatibility
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
                
                quant_model_path = f"{quant_dir}/quantized_int8_model.pth"
                
                torch.save({
                    'model_state_dict': quantized_model.state_dict(),
                    'classes': checkpoint['classes'],
                    'original_accuracy': checkpoint.get('accuracy', 0),
                    'model_name': model_name,
                    'quantization_info': {
                        'method': 'Dynamic INT8',
                        'precision': '8-bit integer',
                        'timestamp': quant_timestamp
                    }
                }, quant_model_path)
                
                method_desc = "Dynamic INT8"
            
            # Calculate file sizes
            original_size_mb = os.path.getsize(model_path) / (1024*1024)
            quantized_size_mb = os.path.getsize(quant_model_path) / (1024*1024)
            compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0
            
            # Save quantization config
            quant_config = {
                'original_model_dir': model_session_dir,
                'quantization_method': method_desc,
                'quantization_type': quantization_type,
                'model_name': model_name,
                'classes': checkpoint['classes'],
                'timestamp': quant_timestamp,
                'original_size_mb': round(original_size_mb, 2),
                'quantized_size_mb': round(quantized_size_mb, 2),
                'compression_ratio': round(compression_ratio, 2),
                'size_reduction_percent': round((1 - quantized_size_mb/original_size_mb) * 100, 1) if original_size_mb > 0 else 0
            }
            
            with open(f"{quant_dir}/quantization_config.json", 'w') as f:
                json.dump(quant_config, f, indent=2)
            
            result_msg = f"✅ Modelo cuantizado ({method_desc})\n"
            result_msg += f"📁 Guardado en: {quant_dir}\n"
            result_msg += f"📊 Tamaño original: {original_size_mb:.1f} MB\n"
            result_msg += f"📊 Tamaño cuantizado: {quantized_size_mb:.1f} MB\n"
            result_msg += f"🔄 Reducción: {quant_config['size_reduction_percent']:.1f}%"
            
            return True, result_msg
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return False, f"❌ Error en cuantización: {str(e)}\n\nDetalles:\n{error_details}"
    
    def convert_model(self, model_session_dir, format_type):
        """Convert trained model to different formats"""
        from datetime import datetime
        
        try:
            # Create conversion directory
            conv_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conv_dir = f"/workspace/models/converted_{format_type}_{conv_timestamp}"
            os.makedirs(conv_dir, exist_ok=True)
            
            # Load the trained model
            model_path = f"{model_session_dir}/model.pth"
            if not os.path.exists(model_path):
                return False, "❌ No se encontró modelo entrenado"
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model_name = checkpoint.get('model_name', 'resnet18')
            num_classes = len(checkpoint['classes'])
            
            # Try to detect model architecture from state_dict if model_name is wrong
            state_dict_keys = list(checkpoint['model_state_dict'].keys())
            if state_dict_keys:
                first_key = state_dict_keys[0]
                # Detect architecture from weight names
                if first_key.startswith('features.'):
                    if 'block.' in first_key:
                        model_name = 'efficientnet_b0'  # Default EfficientNet
                    elif 'inverted_residual' in first_key or 'conv_dw' in first_key:
                        model_name = 'mobilenet_v2'  # Default MobileNet
                elif first_key.startswith('conv1.') or first_key.startswith('layer1.'):
                    model_name = 'resnet18'  # Default ResNet
            
            # Recreate model with detected architecture
            model = self.create_model(num_classes, model_name)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # If loading fails, try to determine the correct architecture
                if 'features.' in str(e) and 'block.' in str(e):
                    # Try different EfficientNet variants
                    for variant in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']:
                        try:
                            model = self.create_model(num_classes, variant)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            model_name = variant
                            break
                        except:
                            continue
                    else:
                        raise RuntimeError(f"Could not load model. Architecture mismatch. Keys suggest EfficientNet but couldn't match variant.")
                elif 'features.' in str(e) and 'classifier.' in str(e):
                    # Try MobileNet variants
                    for variant in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
                        try:
                            model = self.create_model(num_classes, variant)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            model_name = variant
                            break
                        except:
                            continue
                    else:
                        raise RuntimeError(f"Could not load model. Architecture mismatch. Keys suggest MobileNet but couldn't match variant.")
                else:
                    raise e
            
            model.eval()
            
            # Move to CPU for conversion (safer)
            model = model.cpu()
            
            result_files = []
            
            if format_type == "onnx":
                # Convert to ONNX
                dummy_input = torch.randn(1, 3, 224, 224)
                onnx_path = f"{conv_dir}/model.onnx"
                
                torch.onnx.export(
                    model, 
                    dummy_input, 
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}}
                )
                result_files.append(f"model.onnx ({os.path.getsize(onnx_path) / (1024*1024):.1f} MB)")
                
            elif format_type == "torchscript":
                # Convert to TorchScript (traced)
                dummy_input = torch.randn(1, 3, 224, 224)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_path = f"{conv_dir}/model_traced.pt"
                traced_model.save(traced_path)
                result_files.append(f"model_traced.pt ({os.path.getsize(traced_path) / (1024*1024):.1f} MB)")
                
                # Convert to TorchScript (scripted) - may fail for some models
                try:
                    scripted_model = torch.jit.script(model)
                    scripted_path = f"{conv_dir}/model_scripted.pt"
                    scripted_model.save(scripted_path)
                    result_files.append(f"model_scripted.pt ({os.path.getsize(scripted_path) / (1024*1024):.1f} MB)")
                except Exception as script_error:
                    # Scripting might fail for some models, but tracing should work
                    with open(f"{conv_dir}/script_error.txt", 'w') as f:
                        f.write(f"Scripted conversion failed: {str(script_error)}")
                    result_files.append("model_scripted.pt (failed - see script_error.txt)")
                
            elif format_type == "tensorrt":
                # TensorRT conversion via ONNX
                try:
                    # First convert to ONNX
                    dummy_input = torch.randn(1, 3, 224, 224)
                    onnx_path = f"{conv_dir}/model_for_tensorrt.onnx"
                    
                    torch.onnx.export(
                        model, 
                        dummy_input, 
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output']
                    )
                    
                    # Try to convert to TensorRT automatically
                    try:
                        import tensorrt as trt
                        
                        # Build TensorRT engine
                        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                        engine_path = f"{conv_dir}/model.trt"
                        
                        with trt.Builder(TRT_LOGGER) as builder:
                            # Create network with explicit batch
                            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                            network = builder.create_network(network_flags)
                            
                            # Configure builder
                            config = builder.create_builder_config()
                            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
                            
                            # Enable FP16 if available
                            if builder.platform_has_fast_fp16:
                                config.set_flag(trt.BuilderFlag.FP16)
                            
                            # Parse ONNX
                            parser = trt.OnnxParser(network, TRT_LOGGER)
                            with open(onnx_path, 'rb') as model_file:
                                if not parser.parse(model_file.read()):
                                    for error in range(parser.num_errors):
                                        print(parser.get_error(error))
                                    raise RuntimeError("Failed to parse ONNX file")
                            
                            # Build engine
                            engine = builder.build_serialized_network(network, config)
                            if engine is None:
                                raise RuntimeError("Failed to build TensorRT engine")
                            
                            # Save engine
                            with open(engine_path, "wb") as f:
                                f.write(engine)
                            
                            result_files.append(f"model_for_tensorrt.onnx ({os.path.getsize(onnx_path) / (1024*1024):.1f} MB)")
                            result_files.append(f"model.trt ({os.path.getsize(engine_path) / (1024*1024):.1f} MB) - TensorRT Engine")
                            
                    except Exception as trt_error:
                        # If TensorRT conversion fails, create script as fallback
                        trt_script = f"{conv_dir}/convert_to_tensorrt.py"
                        with open(trt_script, 'w') as f:
                            f.write(f'''#!/usr/bin/env python3
# TensorRT Conversion Script

import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path):
    """Build TensorRT engine from ONNX model"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with trt.Builder(TRT_LOGGER) as builder:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_file_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            return None
            
        with open(engine_file_path, "wb") as f:
            f.write(engine)
        return True

if __name__ == "__main__":
    onnx_path = "{onnx_path}"
    engine_path = "{conv_dir}/model.trt"
    
    print("Converting ONNX to TensorRT...")
    success = build_engine(onnx_path, engine_path)
    if success:
        print(f"TensorRT engine saved to: {{engine_path}}")
    else:
        print("Failed to create TensorRT engine")
''')
                        
                        result_files.append(f"model_for_tensorrt.onnx ({os.path.getsize(onnx_path) / (1024*1024):.1f} MB)")
                        result_files.append("convert_to_tensorrt.py (conversion script)")
                        result_files.append(f"⚠️ TensorRT auto-conversion failed: {str(trt_error)[:100]}...")
                        
                        # Add instructions
                        instructions_path = f"{conv_dir}/TENSORRT_INSTRUCTIONS.txt"
                        with open(instructions_path, 'w') as f:
                            f.write("""TensorRT Conversion Instructions:

1. TensorRT is installed but auto-conversion failed
2. Run the conversion script manually:
   python convert_to_tensorrt.py

3. The resulting .trt file can be used for inference with TensorRT

Note: TensorRT requires NVIDIA GPU and specific CUDA/cuDNN versions.
Check error details in the conversion output.
""")
                        result_files.append("TENSORRT_INSTRUCTIONS.txt")
                    
                except Exception as trt_error:
                    return False, f"❌ Error en conversión TensorRT: {str(trt_error)}"
            
            elif format_type == "tflite":
                return False, "❌ Conversión a TensorFlow Lite requiere instalación de tensorflow"
            
            else:
                return False, f"❌ Formato no soportado: {format_type}"
            
            # Save conversion config
            conv_config = {
                'original_model_dir': model_session_dir,
                'conversion_format': format_type,
                'model_name': model_name,
                'classes': checkpoint['classes'],
                'timestamp': conv_timestamp,
                'input_shape': [1, 3, 224, 224],
                'num_classes': num_classes,
                'result_files': result_files
            }
            
            with open(f"{conv_dir}/conversion_config.json", 'w') as f:
                json.dump(conv_config, f, indent=2)
            
            result_msg = f"✅ Modelo convertido a {format_type.upper()}\n"
            result_msg += f"📁 Guardado en: {conv_dir}\n"
            result_msg += f"📄 Archivos generados:\n"
            for file_info in result_files:
                result_msg += f"  • {file_info}\n"
            
            return True, result_msg
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return False, f"❌ Error en conversión: {str(e)}\n\nDetalles:\n{error_details}"

# Initialize trainer
trainer = SimpleTrainer()
temp_data_dir = None

def validate_zip(file):
    global temp_data_dir
    if temp_data_dir and os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)
    
    temp_data_dir, result = trainer.extract_and_validate_zip(file)
    return result

def get_model_info(model_name):
    """Get detailed information about selected model"""
    models_info = trainer.get_available_models()
    if model_name in models_info:
        info = models_info[model_name]
        result = f"🤖 **{info['name']}**\n\n"
        result += f"📝 {info['description']}\n\n"
        result += f"**📊 Especificaciones:**\n"
        result += f"• {info['params']}\n"
        result += f"• Tamaño: {info['size']}\n"
        result += f"• Velocidad: {info['speed']}\n"
        result += f"• Precisión esperada: {info['accuracy']}\n\n"
        result += f"**⚡ Cuantización INT8 (Edge/Mobile):**\n"
        result += f"• Tamaño optimizado: {info['edge_quantized']}\n\n"
        result += f"**🎯 Caso de uso ideal:**\n"
        result += f"• {info['use_case']}"
        return result
    return "Selecciona un modelo para ver información detallada"

def train_model(epochs, learning_rate, model_name, 
                use_augmentation, rotation_enabled, rotation_degrees,
                zoom_enabled, zoom_range, brightness_enabled, brightness_range,
                flip_horizontal, flip_vertical, progress=gr.Progress()):
    global temp_data_dir
    
    if temp_data_dir is None or not os.path.exists(temp_data_dir):
        return "❌ Primero valida un dataset", None
    
    # Store selected model in trainer
    trainer.selected_model = model_name
    
    def progress_callback(info):
        # Update Gradio progress bar with detailed info
        prog = trainer.get_training_progress()
        detailed_desc = f"Época {prog['current_epoch']}/{prog['total_epochs']} | Batch {prog['current_batch']}/{prog['total_batches']} | {info}"
        progress(prog['overall_progress'], desc=detailed_desc)
    
    try:
        epochs = int(epochs)
        if epochs < 1 or epochs > 40:
            return "❌ Épocas debe estar entre 1 y 40", None
        
        models_info = trainer.get_available_models()
        model_info = models_info.get(model_name, {})
        progress(0, desc=f"🚀 Iniciando entrenamiento con {model_info.get('name', model_name)}...")
        
        # Create augmentation config to pass to trainer
        augmentation_config = {
            'use_augmentation': use_augmentation,
            'rotation_enabled': rotation_enabled,
            'rotation_degrees': rotation_degrees,
            'zoom_enabled': zoom_enabled,
            'zoom_range': zoom_range,
            'brightness_enabled': brightness_enabled,
            'brightness_range': brightness_range,
            'flip_horizontal': flip_horizontal,
            'flip_vertical': flip_vertical
        }
        
        history = trainer.train_model(
            temp_data_dir, 
            epochs=epochs, 
            lr=learning_rate,
            progress_callback=progress_callback,
            augmentation_config=augmentation_config
        )
        
        progress(1.0, desc="✅ Entrenamiento completado!")
        
        result = f"✅ **Entrenamiento completado exitosamente!**\n\n"
        result += f"🤖 **Modelo utilizado:** {model_info.get('name', model_name)}\n"
        result += f"📊 **Configuración:**\n"
        result += f"• Épocas entrenadas: {epochs}\n"
        result += f"• Learning rate: {learning_rate}\n"
        result += f"• Parámetros del modelo: {model_info.get('params', 'N/A')}\n"
        result += f"• Tamaño estimado: {model_info.get('size', 'N/A')}\n"
        result += f"• Clases detectadas: {len(trainer.classes)}\n"
        result += f"• Clases: {', '.join(trainer.classes)}\n\n"
        
        if history:
            result += f"📈 **Métricas finales:**\n"
            result += f"• Pérdida final (entrenamiento): {history['train_loss'][-1]:.4f}\n"
            result += f"• Pérdida final (validación): {history['val_loss'][-1]:.4f}\n"
            result += f"• Precisión final (entrenamiento): {history['train_acc'][-1]:.2f}%\n"
            result += f"• Precisión final (validación): {history['val_acc'][-1]:.2f}%\n\n"
            
            # Best metrics
            best_val_acc = max(history['val_acc'])
            best_val_epoch = history['epochs'][history['val_acc'].index(best_val_acc)]
            result += f"🏆 **Mejor resultado:**\n"
            result += f"• Mejor precisión validación: {best_val_acc:.2f}% (época {best_val_epoch})\n"
            result += f"\n📊 Gráficas y matriz de confusión generadas correctamente"
        
        # Load plots
        plots = trainer.get_training_plots()
        
        return result, plots
        
    except Exception as e:
        return f"❌ Error durante entrenamiento: {str(e)}", None


def predict_image(image):
    return trainer.predict(image)

def show_model_info():
    return trainer.get_model_info()

def get_available_trained_models():
    """Get list of available trained models from timestamped directories"""
    models_dir = "/workspace/models"
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    for item in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, item)
        if os.path.isdir(model_dir) and not item.startswith('quantized_') and not item.startswith('converted_'):
            model_path = os.path.join(model_dir, 'model.pth')
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(model_path):
                try:
                    # Get model info from config
                    model_info = f"{item}"
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            model_name = config.get('model_name', 'unknown')
                            accuracy = config.get('final_metrics', {}).get('val_acc', 0)
                            model_info = f"{item} - {model_name} (Acc: {accuracy:.1f}%)"
                    available_models.append((model_info, model_dir))
                except:
                    available_models.append((item, model_dir))
    
    return sorted(available_models, reverse=True)  # Most recent first

def quantize_selected_model(model_selection, quant_type):
    """Quantize selected model"""
    if not model_selection:
        return "❌ Selecciona un modelo para cuantizar"
    if not quant_type:
        return "❌ Selecciona un tipo de cuantización"
    
    success, message = trainer.quantize_model(model_selection, quant_type)
    return message

def convert_selected_model(model_selection, format_type):
    """Convert selected model to specified format"""
    if not model_selection:
        return "❌ Selecciona un modelo para convertir"
    if not format_type:
        return "❌ Selecciona un formato de conversión"
    
    success, message = trainer.convert_model(model_selection, format_type)
    return message

def load_training_plots(model_dir=None):
    if model_dir:
        # Load specific model's plots and metrics
        plots_path = f"{model_dir}/training_plots.png"
        history_path = f"{model_dir}/training_history.json"
        config_path = f"{model_dir}/config.json"
        
        plots = None
        if os.path.exists(plots_path):
            try:
                from PIL import Image
                plots = Image.open(plots_path)
            except:
                plots = None
        
        if plots is None:
            return None, "❌ No hay gráficas disponibles para este modelo."
        
        summary = f"📊 **Resumen del Entrenamiento**\n\n"
        
        # Load model config for dataset info
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get('model_name', 'Desconocido')
                    classes = config.get('classes', [])
                    summary += f"• **Modelo**: {model_name}\n"
                    summary += f"• **Dataset**: {len(classes)} clases\n"
                    summary += f"• **Clases**: {', '.join(classes)}\n\n"
            except:
                pass
        
        # Load training metrics
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    metrics = json.load(f)
                    summary += f"• **Total épocas**: {len(metrics.get('epochs', []))}\n"
                    train_acc = metrics.get('train_acc', [])
                    val_acc = metrics.get('val_acc', [])
                    train_loss = metrics.get('train_loss', [])
                    val_loss = metrics.get('val_loss', [])
                    
                    if train_acc and val_acc:
                        summary += f"• **Mejor precisión (train)**: {max(train_acc):.2f}%\n"
                        summary += f"• **Mejor precisión (val)**: {max(val_acc):.2f}%\n"
                        summary += f"• **Menor pérdida (train)**: {min(train_loss):.4f}\n"
                        summary += f"• **Menor pérdida (val)**: {min(val_loss):.4f}\n"
                        
                        # Check for overfitting
                        final_train_acc = train_acc[-1]
                        final_val_acc = val_acc[-1]
                        if final_train_acc - final_val_acc > 10:
                            summary += f"\n⚠️ **Advertencia**: Posible sobreajuste (diferencia: {final_train_acc - final_val_acc:.1f}%)"
            except:
                summary += "• Error cargando métricas de entrenamiento\n"
        
        return plots, summary
    else:
        # Default behavior for current trainer model
        plots = trainer.get_training_plots()
        if plots is None:
            return None, "❌ No hay gráficas disponibles. Entrena un modelo primero."
        
        metrics = trainer.get_metrics_data()
        if metrics:
            summary = f"📊 **Resumen del Entrenamiento**\n\n"
            summary += f"• Total épocas: {len(metrics['epochs'])}\n"
            summary += f"• Mejor precisión (train): {max(metrics['train_acc']):.2f}%\n"
            summary += f"• Mejor precisión (val): {max(metrics['val_acc']):.2f}%\n"
            summary += f"• Menor pérdida (train): {min(metrics['train_loss']):.4f}\n"
            summary += f"• Menor pérdida (val): {min(metrics['val_loss']):.4f}\n"
            
            # Check for overfitting
            final_train_acc = metrics['train_acc'][-1]
            final_val_acc = metrics['val_acc'][-1]
            if final_train_acc - final_val_acc > 10:
                summary += f"\n⚠️ **Advertencia:** Posible sobreajuste detectado (diferencia: {final_train_acc - final_val_acc:.1f}%)"
            
            return plots, summary
    
    return plots, "📊 Gráficas de entrenamiento"

# Gradio Interface with Rachael.vision theme
with gr.Blocks(title=t('title'), theme=gr.themes.Soft().set(
    background_fill_primary='#f8f9fa',
    background_fill_secondary='#e9ecef', 
    block_background_fill='#ffffff',
    body_text_color='#212529',
    block_label_text_color='#495057',
    button_primary_background_fill='#0d6efd',
    button_primary_background_fill_hover='#0b5ed7',
    button_secondary_background_fill='#6c757d',
    button_secondary_background_fill_hover='#5c636a'
)) as demo:
    gr.Markdown(f"# {t('title')}")
    if CURRENT_LANGUAGE == 'en':
        gr.Markdown(f"### {t('subtitle_professional_en')}")
        gr.Markdown(t('subtitle_advanced_en'))
    else:
        gr.Markdown(f"### {t('subtitle_professional_es')}")
        gr.Markdown(t('subtitle_advanced_es'))
    
    with gr.Tab(f"📁 {t('dataset_tab')}"):
        gr.Markdown(f"### {t('upload_validate_title')}")
        gr.Markdown(t('upload_validate_desc'))
        
        zip_input = gr.File(label=t('upload_zip'), file_types=[".zip"])
        validate_btn = gr.Button(f"🔍 {t('validate_button')}", variant="primary")
        validation_output = gr.Textbox(label=f"✅ {t('validation_result')}", lines=8)
        
        validate_btn.click(validate_zip, inputs=zip_input, outputs=validation_output)
    
    with gr.Tab(t('training_tab')):
        gr.Markdown(f"### {t('config_training_desc')}")
        
        # Model selection section
        gr.Markdown(f"#### {t('model_selection')}")
        with gr.Row():
            model_choices = list(trainer.get_available_models().keys())
            model_names = [trainer.get_available_models()[key]["name"] for key in model_choices]
            model_dropdown = gr.Dropdown(
                choices=list(zip(model_names, model_choices)),
                value="resnet18",
                label=t('base_model_label'),
                info=t('base_model_info')
            )
            
        model_info_display = gr.Textbox(
            label=t('model_info_label'),
            lines=12,
            value=get_model_info("resnet18"),
            interactive=False
        )
        
        # Training parameters
        gr.Markdown(f"#### {t('training_parameters_title')}")
        with gr.Row():
            epochs_input = gr.Slider(1, 40, value=5, step=1, label=t('epochs_label'))
            lr_input = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label=f"📈 {t('learning_rate')}")
        
        # Data Augmentation parameters
        gr.Markdown(f"#### {t('data_augmentation_title')}")
        with gr.Row():
            use_augmentation = gr.Checkbox(label=t('use_data_augmentation'), value=True)
        
        with gr.Row():
            with gr.Column():
                rotation_enabled = gr.Checkbox(label=t('rotation_label'), value=True)
                rotation_degrees = gr.Slider(0, 45, value=15, label=t('max_degrees_label'), interactive=True)
                
                zoom_enabled = gr.Checkbox(label=t('zoom_label'), value=True)
                zoom_range = gr.Slider(0.7, 1.3, value=0.9, label=t('zoom_range_label'), interactive=True)
                
            with gr.Column():
                brightness_enabled = gr.Checkbox(label=t('brightness_contrast_label'), value=True)
                brightness_range = gr.Slider(0.5, 1.5, value=0.8, label=t('brightness_range_label'), interactive=True)
                
                flip_horizontal = gr.Checkbox(label=t('horizontal_flip_label'), value=True)
                flip_vertical = gr.Checkbox(label=t('vertical_flip_label'), value=False)
        
        train_btn = gr.Button(t('start_training_button'), variant="primary")
        
        training_output = gr.Textbox(label=t('training_result_label'), lines=10)
        training_plots = gr.Image(label=t('training_plots_label'))
        
        # Update model info when selection changes
        model_dropdown.change(
            get_model_info,
            inputs=model_dropdown,
            outputs=model_info_display
        )
        
        train_btn.click(
            train_model, 
            inputs=[
                epochs_input, lr_input, model_dropdown,
                use_augmentation, rotation_enabled, rotation_degrees,
                zoom_enabled, zoom_range, brightness_enabled, brightness_range,
                flip_horizontal, flip_vertical
            ], 
            outputs=[training_output, training_plots]
        )
    
    with gr.Tab(t('metrics_tab_title')):
        gr.Markdown(f"### {t('metrics_training_desc')}")
        
        with gr.Row():
            metrics_model_dropdown = gr.Dropdown(
                choices=[],
                label=t('trained_model_dropdown'),
                info=t('model_metrics_info')
            )
            refresh_metrics_btn = gr.Button(t('refresh_button'), size="sm")
        
        metrics_btn = gr.Button(t('load_metrics_button'), variant="secondary")
        training_plots_display = gr.Image(label=t('training_plots_label'))
        metrics_summary = gr.Textbox(label=t('metrics_summary_label'), lines=8)
        
        def refresh_metrics_models():
            models = get_available_trained_models()
            return gr.Dropdown(choices=models)
        
        def load_selected_metrics(model_selection):
            if not model_selection:
                return None, t('select_trained_model')
            return load_training_plots(model_selection)
        
        refresh_metrics_btn.click(
            refresh_metrics_models,
            outputs=metrics_model_dropdown
        )
        
        metrics_btn.click(
            load_selected_metrics,
            inputs=metrics_model_dropdown,
            outputs=[training_plots_display, metrics_summary]
        )
    
    with gr.Tab(t('quantization_tab_title')):
        gr.Markdown(f"### {t('quantization_title')}")
        gr.Markdown(t('quantization_desc'))
        
        with gr.Row():
            quant_model_dropdown = gr.Dropdown(
                choices=[],
                label=t('model_to_quantize_dropdown'),
                info=t('model_quantize_info')
            )
            refresh_quant_btn = gr.Button(t('refresh_button'), size="sm")
        
        quant_type_dropdown = gr.Dropdown(
            choices=[
                (t('int8_option'), "int8"),
                (t('fp16_option'), "fp16")
            ],
            value="int8",
            label=t('quantization_type_label'),
            info=t('quantization_type_info')
        )
        
        quantize_btn = gr.Button(t('quantize_model_button'), variant="primary")
        quantization_output = gr.Textbox(label=t('quantization_result_label'), lines=8)
        
        def refresh_quantization_models():
            models = get_available_trained_models()
            return gr.Dropdown(choices=models)
        
        refresh_quant_btn.click(
            refresh_quantization_models,
            outputs=quant_model_dropdown
        )
        
        quantize_btn.click(
            quantize_selected_model,
            inputs=[quant_model_dropdown, quant_type_dropdown],
            outputs=quantization_output
        )
    
    with gr.Tab(t('conversion_tab_title')):
        gr.Markdown(f"### {t('conversion_title')}")
        gr.Markdown(t('conversion_desc'))
        
        with gr.Row():
            conv_model_dropdown = gr.Dropdown(
                choices=[],
                label=t('model_to_convert_dropdown'),
                info=t('model_convert_info')
            )
            refresh_conv_btn = gr.Button(t('refresh_button'), size="sm")
        
        format_dropdown = gr.Dropdown(
            choices=[
                (t('onnx_option'), "onnx"),
                (t('torchscript_option'), "torchscript"),
                (t('tensorrt_option'), "tensorrt")
            ],
            value="onnx",
            label=t('target_format_label'),
            info=t('target_format_info')
        )
        
        convert_btn = gr.Button(t('convert_model_button'), variant="primary")
        conversion_output = gr.Textbox(label=t('conversion_result_label'), lines=5)
        
        def refresh_conversion_models():
            models = get_available_trained_models()
            return gr.Dropdown(choices=models)
        
        refresh_conv_btn.click(
            refresh_conversion_models,
            outputs=conv_model_dropdown
        )
        
        convert_btn.click(
            convert_selected_model,
            inputs=[conv_model_dropdown, format_dropdown],
            outputs=conversion_output
        )
    
    with gr.Tab(t('prediction_tab_title')):
        gr.Markdown(f"### {t('prediction_title')}")
        gr.Markdown(t('prediction_desc'))
        
        with gr.Row():
            prediction_model_dropdown = gr.Dropdown(
                choices=[],
                label=t('prediction_model_dropdown'),
                info=t('prediction_model_info')
            )
            refresh_prediction_btn = gr.Button(t('refresh_button'), size="sm")
        
        image_input = gr.Image(type="pil", label=t('image_to_classify_label'))
        predict_btn = gr.Button(t('classify_button'), variant="primary")
        prediction_output = gr.Textbox(label=t('analysis_result_label'), lines=15)
        
        def refresh_prediction_models():
            models = get_available_trained_models()
            return gr.Dropdown(choices=models)
        
        def predict_with_selected_model(image, model_selection):
            if not model_selection:
                return t('select_model_first')
            if image is None:
                return t('upload_image_first')
            
            # Load the selected model
            trainer.model_path = f"{model_selection}/model.pth"
            trainer.model = None  # Reset current model to force reload
            return trainer.predict(image)
        
        refresh_prediction_btn.click(
            refresh_prediction_models,
            outputs=prediction_model_dropdown
        )
        
        predict_btn.click(
            predict_with_selected_model, 
            inputs=[image_input, prediction_model_dropdown], 
            outputs=prediction_output
        )
    
    gr.Markdown("---")
    gr.Markdown(f"## {t('rachael_platform_title')}")
    gr.Markdown(f"### {t('professional_solutions')}")
    gr.Markdown(f"{t('gpu_status')} " + (t('gpu_enabled') if torch.cuda.is_available() else t('cpu_mode')))
    gr.Markdown(f"{t('features_title')} {t('features_list')}")
    gr.Markdown(f"{t('more_info')} [rachael.vision](https://rachael.vision) | {t('version_info')}")
    gr.Markdown(t('copyright_text'))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)