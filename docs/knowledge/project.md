# Project: MSI-Net Visual Saliency for Advertising

Predict where humans look in images, with models fine-tuned for advertising applications.

## Purpose

This project produces visual saliency models (MSI-Net architecture) that predict human fixation patterns in images, specifically optimized for advertising use cases. The model identifies which areas of ad creatives attract visual attention, enabling data-driven optimization of ad design.

Based on the 2020 Neural Networks paper "Contextual encoder-decoder network for visual saliency prediction", this implementation supports multi-platform training (CPU, NVIDIA GPU, Apple Silicon, Google TPU) and deployment for internal ML team experimentation.

## Key Concepts

- **Visual Saliency**: Predicting spatial distribution of human fixations in images using deep learning
- **Multi-Scale Features**: Extracting contextual information at different spatial scales via dilated convolutions
- **Ad Creative Elements**: Focus on attention patterns for CTAs, product placement, brand logos
- **Dataset Types**: Training and evaluation across banner ads, video ads, social media ad formats
- **Performance Metrics**: Attention heatmaps, fixation point prediction, engagement correlation analysis

## Users

Internal ML engineers and data scientists developing and fine-tuning saliency models for advertising optimization. Users train models on advertising-specific datasets and evaluate performance for creative testing applications.
