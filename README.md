# Skin Lesion Segmentation — U-Net

Pixel-level segmentation of skin lesions from dermoscopy images using a U-Net
architecture trained from scratch on the HAM10000 dataset.

## Results

| Metric | Value |
|--------|-------|
| Test Dice Score | 0.9115 |
| Best Val Dice | 0.9191 |
| Best Val Loss | 0.1086 |
| Training Epochs | 20 |

## Architecture

U-Net built from scratch in PyTorch:

- Encoder: 4 blocks (64, 128, 256, 512 channels) with MaxPool
- Bottleneck: 1024 channels
- Decoder: 4 blocks with transposed convolutions and skip connections
- Output: 1x1 conv producing binary segmentation mask
- Total parameters: 31,037,633

## Dataset

- HAM10000 — 10,015 dermoscopy image and mask pairs
- Split: 8,012 train / 1,001 val / 1,002 test
- Input size: 256x256
- Mask format: binary (lesion=1, background=0)

## Training Details

- Loss: Combined BCE + Dice loss (50/50 weight)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Batch size: 16
- Hardware: T4 GPU — Google Colab Free

## Key Features

- U-Net built from scratch — no pretrained backbone
- Combined BCE + Dice loss for handling class imbalance
- Skip connections for precise lesion localisation
- Data augmentation: flips, rotations, colour jitter
- Training curves and segmentation visualisations included

## Tech Stack

- PyTorch
- Albumentations
- OpenCV
- Matplotlib
- Google Colab T4 GPU

## Portfolio Links

- Project 6 YOLOv8 Detector: https://github.com/Boatengs/yolo-object-detector
- Project 5 Skin Disease Classifier: https://github.com/Boatengs/skin-disease-classifier
- Project 4 LLM Eval Framework: https://github.com/Boatengs/llm-evaluation-framework
- Project 3 Medical QA LoRA: https://github.com/Boatengs/medical-qa-lora
- Project 2 SPORTZBOT RAG: https://github.com/Boatengs/sports-rag-chatbot-
- Project 1 Sentiment Analyzer: https://github.com/Boatengs/sentiment-analyzer
