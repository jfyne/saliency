# Tech Stack

## Primary Language

**Python 3.10+** (supports 3.10, 3.11, 3.12)
- Python 3.11 required for Apple Metal GPU support (tensorflow-metal compatibility)

## Deep Learning Framework

**TensorFlow 2.15+**
- Core framework for model training and inference
- TensorFlow ≤2.18.1 constraint for Apple Metal compatibility

## Package Management

**uv** (recommended)
- Modern Python package manager with lock file support (`uv.lock`)
- Fallback: pip with `pyproject.toml`

## Multi-Platform Accelerators

### NVIDIA GPU (CUDA)
- Extra: `tensorflow[and-cuda]>=2.15.0`
- Data format: `channels_first`
- Config: `device: "gpu"`

### Apple Silicon (Metal)
- Extra: `tensorflow-metal>=1.0.0` with `tensorflow<=2.18.1`
- Data format: `channels_last`
- Config: `device: "metal"`
- Requires Python 3.11, macOS 12.0+

### Google Cloud TPU
- TPU v5e for cost-effective training (~$1.20/chip/hour)
- Data format: `channels_last`
- Config: `device: "tpu"`

### CPU
- Data format: `channels_last`
- Config: `device: "cpu"`

## Cloud Infrastructure

**Google Cloud Vertex AI**
- Custom training jobs with GPU/TPU
- Model registry for versioning
- Containerized training (Dockerfile, Dockerfile.tpu)

## Build Automation

**just** (justfile)
- Local training commands
- Cloud training orchestration
- Container build and push
- Model registry operations

## Code Quality

**Ruff**
- Linting and formatting
- Line length: 100
- Target: Python 3.10+

## Related Topics

- [Apple Silicon](apple-silicon.md) - Detailed Metal GPU setup and MLX alternative
- [Cloud Training](cloud-training.md) - Vertex AI deployment details
- [Dependencies](dependencies.md) - Full dependency specifications
- [Weight Format Compatibility](weight-format-compatibility.md) - Cross-platform considerations
