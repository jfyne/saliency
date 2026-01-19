# Dependencies

## Core Dependencies

### Deep Learning
- **tensorflow>=2.15.0** - Deep learning framework for MSI-Net architecture
  - Version constraint: ≤2.18.1 when using Metal GPU
  - Provides Keras API for model building

### Numerical Computing
- **numpy>=1.26.0** - Array operations and numerical processing
- **scipy>=1.11.0** - Scientific computing (Gaussian filtering, image operations)

### Image Processing
- **imageio>=2.33.0** - Image file I/O (supports various formats)
- **h5py>=3.10.0** - HDF5 file format support (model weights, large datasets)

### Visualization
- **matplotlib>=3.8.0** - Plotting and saliency map visualization

### Data Acquisition
- **requests>=2.31.0** - HTTP downloads for datasets
- **gdown>=5.0.0** - Google Drive downloads (MIT1003, SALICON via Drive)

## Optional Dependencies

### GPU Acceleration (NVIDIA)
- **tensorflow[and-cuda]>=2.15.0** - CUDA support for NVIDIA GPUs
  - Includes: nvidia-cudnn-cu12, nvidia-cuda-cupti-cu12, etc.
  - Use: `uv sync --extra gpu`
  - Data format: channels_first (NCHW)

### GPU Acceleration (Apple Silicon)
- **tensorflow>=2.15.0,<=2.18.1** - Version-constrained TensorFlow
- **tensorflow-metal>=1.0.0** - Metal GPU plugin for M1/M2/M3
  - Requirements: Python ≤3.11, macOS 12.0+
  - Use: `uv sync --extra metal`
  - Data format: channels_last (NHWC)

### Development Tools
- **ruff>=0.1.0** - Fast Python linter and formatter
  - Config: line-length=100, target=py310
  - Enabled rules: E (errors), F (pyflakes), I (isort), W (warnings)
- **pytest>=7.0.0** - Testing framework (configured but no tests yet)
  - Use: `uv sync --extra dev`

## External Services

### Google Cloud Platform
- **Vertex AI** - Cloud training and model registry
  - Custom training jobs with GPU (NVIDIA A100, T4)
  - TPU support (v5e for cost-effective training)
  - Model versioning and artifact storage
  - Base image: `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest`

### Cloud Storage
- **Google Cloud Storage (GCS)** - Model artifact storage
  - Environment variable: `GCS_OUTPUT_PATH`
  - Format: `gs://bucket-name/path/to/models/{model}`
  - Used for Vertex AI training outputs

### Container Registry
- **Google Artifact Registry** - Docker image hosting
  - Repository: `us-central1-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/saliency`
  - Images: training containers with all dependencies pre-installed

## Datasets (Auto-Downloaded)

Training and evaluation datasets managed by `download.py`:
- **SALICON** - 10,000 training + 5,000 validation images
- **MIT1003** - 1,003 images with eye-tracking
- **CAT2000** - 2,000 categorized images
- **DUTOMRON** - Object-based saliency dataset
- **PASCALS** - PASCAL-S dataset
- **OSIE** - Object and semantic images
- **FIWI** - Free-viewing dataset
- **FIXATIONADD1000** - Additional fixation dataset (1,000 images)

## Related Topics

- [Dataset System](dataset-system.md) - Adding new datasets
- [Tech Stack](tech-stack.md) - Platform-specific requirements
- [Apple Silicon](apple-silicon.md) - tensorflow-metal installation details
