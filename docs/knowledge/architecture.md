# Architecture

## Directory Structure

```
saliency/
├── main.py              # CLI entry point (train/test commands)
├── config.py            # Training parameters and dataset dimensions
├── model.py             # MSI-Net architecture (encoder-decoder + multi-scale module)
├── data.py              # Dataset loading and tf.data pipelines
├── download.py          # Dataset download utilities (SALICON, MIT1003, etc.)
├── utils.py             # Image processing and visualization helpers
├── demo/                # TensorFlow.js web demo
├── docs/
│   ├── knowledge/       # Project context and conventions
│   ├── plans/           # Implementation plans
│   └── research/        # Research and investigation documents
├── figures/             # Documentation images (architecture, results)
├── results/
│   ├── ckpts/           # Model checkpoints (.weights.h5 files and SavedModel directories)
│   └── history/         # Training history (.txt files and curve plots)
├── Dockerfile           # Vertex AI GPU training container
├── Dockerfile.tpu       # Vertex AI TPU training container
├── justfile             # Build automation and training commands
└── vertex-ai-config.yaml # Google Cloud training configuration
```

## Entry Points

### CLI (main.py)
- `python main.py train -d <dataset> -p <path>` - Train model on dataset
- `python main.py test -d <dataset> -p <path>` - Generate saliency maps

Command-line arguments:
- `-d/--dataset`: salicon, mit1003, cat2000, dutomron, pascals, osie, fiwi, fixationadd1000
- `-p/--path`: Data directory path (default: `data/`)

### Justfile Commands
- `just train-salicon` - Local SALICON training
- `just train-salicon-metal` - Apple Metal GPU training
- `just train-vertex-salicon` - Vertex AI GPU training
- `just train-vertex-salicon-tpu` - Vertex AI TPU training
- `just build-container` - Build and push Docker image
- `just upload-model` - Upload to Vertex AI Model Registry

## Key Modules

### model.py (401 lines)
MSI-Net architecture implementation:
- Encoder: VGG16 backbone (pre-trained on ImageNet)
- Multi-scale module: ASPP with parallel dilated convolutions (rates: 4, 8, 12) plus rate 2 in encoder conv5
- Decoder: Bilinear upsampling with skip connections
- Device-aware optimization (channels_first for GPU, channels_last for CPU/TPU/Metal)

### data.py (832 lines)
Data pipeline management:
- `load_data_training()`: tf.data pipeline with augmentation
- `load_data_validation()`: Validation set loading
- `load_data_testing()`: Test set with arbitrary image sizes
- Dataset-specific preprocessing for each benchmark

### download.py (499 lines)
Automated dataset acquisition:
- Google Drive downloads via gdown
- Direct HTTP downloads for MIT1003, CAT2000, etc.
- Extraction and directory setup
- Progress tracking

### utils.py (253 lines)
Helper functions:
- `postprocess_saliency()`: Convert predictions to heatmaps
- `save_saliency()`: Export results with original dimensions
- Gaussian blur operations for saliency smoothing
- Device detection and configuration helpers

### config.py (50 lines)
Centralized configuration:
- `PARAMS`: Training hyperparameters (epochs, batch_size, learning_rate, device)
- `DIMS`: Dataset-specific image dimensions (must be divisible by 8)
- `GCS_OUTPUT_PATH`: Cloud storage path for model artifacts

## Related Topics

- [Dataset System](dataset-system.md) - Adding and managing training datasets
- [Cloud Training](cloud-training.md) - Vertex AI GPU/TPU training
- [Apple Silicon](apple-silicon.md) - Metal GPU training on Mac
- [Weight Format Compatibility](weight-format-compatibility.md) - Cross-platform weight portability
