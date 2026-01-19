# Cloud Training (Vertex AI)

Running MSI-Net training on Google Cloud Vertex AI with GPU and TPU accelerators.

## Overview

The codebase supports cloud training via Google Cloud Vertex AI using containerized training jobs. GPU training works out of the box with `channels_first` format, while TPU training requires `channels_last` format (using the `tpu` device setting). Model artifacts can be stored in Google Cloud Storage (GCS) and uploaded to Vertex AI Model Registry.

## GPU vs TPU Training

| Aspect | GPU | TPU |
|--------|-----|-----|
| Data Format | `channels_first` (NCHW) | `channels_last` (NHWC) |
| Device Config | `device: "gpu"` | `device: "tpu"` |
| Code Changes | None required | Use existing `tpu` device option |
| Container | `Dockerfile` | `Dockerfile.tpu` |
| Accelerator Types | T4, V100, A100, H100 | v5e (1, 4, 8 chips) |

## Container Configuration

**GPU Container (Dockerfile):**
```dockerfile
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py", "train", "-d", "salicon"]
```

**TPU Container (Dockerfile.tpu):**
- Base: Python 3.10 slim
- Install TensorFlow TPU wheel from GCS
- Set `PJRT_DEVICE=TPU` environment variable

## GCS Output Path

Model artifacts can be written directly to GCS using the `GCS_OUTPUT_PATH` environment variable:

```bash
export GCS_OUTPUT_PATH=gs://your-bucket/models/salicon/v1/
```

When set, all outputs are written to GCS:
- Checkpoints: `gs://bucket/ckpts/best/model_{dataset}_{device}.weights.h5`
- SavedModel: `gs://bucket/ckpts/best/model_{dataset}_{device}/`
- History: `gs://bucket/history/train_{dataset}_{device}.txt`

The codebase uses `tf.io.gfile` for transparent GCS/local filesystem access.

## Vertex AI Machine Types

**GPU Options:**

| Machine Type | GPU | Use Case |
|-------------|-----|----------|
| n1-standard-4 + NVIDIA_TESLA_T4 | 1x T4 | Cost-effective |
| n1-standard-8 + NVIDIA_TESLA_V100 | 1x V100 | Balanced |
| a2-highgpu-1g + NVIDIA_A100 | 1x A100 | High performance |

**TPU Options:**
- `ct5lp-hightpu-1t` (1 chip, ~$1.20/hour)
- `ct5lp-hightpu-4t` (4 chips)
- `ct5lp-hightpu-8t` (8 chips)

## Training Commands

**Build and push container:**
```bash
just build-container
```

**Submit GPU training job:**
```bash
just train-vertex-salicon
```

**Submit TPU training job:**
```bash
just train-vertex-salicon-tpu
```

## Model Registry Upload

After training, upload the SavedModel to Vertex AI Model Registry:

```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=msi-net-salicon \
  --artifact-uri=gs://your-bucket/ckpts/best/model_salicon_gpu/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-15:latest
```

Or use the justfile command:
```bash
just upload-model
```

## GCS Bucket Structure

Recommended organization for model artifacts:

```
gs://your-bucket/
  models/
    salicon/
      v1/
        model_salicon_gpu.weights.h5
        model_salicon_gpu/  # SavedModel directory
      v2/
        ...
    fixationadd1000/
      v1/
        ...
  datasets/
    salicon/
    fixationadd1000/
```

## Configuration File

The `vertex-ai-config.yaml` file defines training job specifications:
- Machine type and GPU/TPU count
- Container image URI
- Environment variables (GCS_OUTPUT_PATH)
- Dataset and hyperparameter settings

## Related Topics

- [Architecture](architecture.md) - Model export and SavedModel format
- [Weight Format Compatibility](weight-format-compatibility.md) - GPU vs TPU weight compatibility
- [Tech Stack](tech-stack.md) - TensorFlow version requirements

## Source Documents

- `docs/research/2025-12-10-vertex-ai-training.md` - Vertex AI compatibility research
- `docs/plans/vertex-ai-training.md` - Implementation plan with task list
