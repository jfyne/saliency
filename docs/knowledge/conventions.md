# Code Conventions

## File Naming

### Python Modules
- Snake_case for module names: `data.py`, `model.py`, `utils.py`, `download.py`
- No package subdirectories - flat structure at root level
- Single-word names preferred for clarity

### Checkpoints and Results
- Model weights: `results/ckpts/{best,latest}/model_{dataset}_{device}.weights.h5`
- SavedModel export: `results/ckpts/best/model_{dataset}_{device}/` (directory)
- Training history: `results/history/train_{dataset}_{device}.txt`, `valid_{dataset}_{device}.txt`
- Training curves: `results/history/curve_{dataset}_{device}.png`
- Generated saliency maps: `results/images/{dataset}/`

### Documentation
- Research docs: `docs/research/YYYY-MM-DD-topic.md` (ISO date prefix)
- Implementation plans: `docs/plans/topic.md` (no date)
- Knowledge base: `docs/knowledge/topic.md`

## Test Patterns

**Current Status**: No tests implemented yet

**Future Conventions** (when tests are added):
- Test files: `test_<module>.py` (e.g., `test_model.py`, `test_data.py`)
- Test location: Root level or `tests/` directory
- Test framework: pytest
- Test discovery: `pytest` (auto-discovers `test_*.py` files)

## Import Organization

### Import Order
1. Standard library imports
2. Third-party imports (numpy, tensorflow, etc.)
3. Local module imports

### Example Pattern
```python
import os
import sys

import numpy as np
import tensorflow as tf

import config
import download
import model
```

### Import Style
- Absolute imports for local modules: `import config` (not `from config import PARAMS`)
- Access via module namespace: `config.PARAMS["device"]`
- Selective imports for frequently used functions acceptable: `from utils import postprocess_saliency`

## Class and Function Patterns

### Class Definitions
- PascalCase for class names: `MSINET`, `SALICON`, `MIT1003`
- Dataset classes as class constants: `n_train`, `n_valid`
- Keras Model subclasses use `tf.keras.Model` base
- Private attributes use single underscore: `self._data_format`, `self._channel_axis`

### Function Definitions
- Snake_case for function names: `kld_loss()`, `load_data()`, `define_paths()`
- Comprehensive docstrings with Args/Returns sections
- Google-style docstrings preferred

### Docstring Format
```python
def function_name(arg1, arg2):
    """Brief description of function purpose.

    Args:
        arg1 (type): Description of arg1.
        arg2 (type, optional): Description of arg2. Defaults to value.

    Returns:
        type: Description of return value.
    """
```

## Error Handling

### Current Patterns
- Automatic dataset downloads when data paths don't exist
- Graceful fallbacks with print warnings (e.g., Metal GPU detection)
- Environment variable checks with defaults: `os.environ.get("GCS_OUTPUT_PATH", None)`
- TensorFlow error suppression: `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"`

### Device-Specific Behavior
- Check `config.PARAMS["device"]` for platform-specific logic
- Set data format based on device: `channels_first` for GPU, `channels_last` for CPU/TPU/Metal
- Enable GPU memory growth to prevent Metal segfaults

## Configuration Management

### config.py Structure
- Global dictionaries for settings: `PARAMS`, `DIMS`
- Environment variable integration: `GCS_OUTPUT_PATH`
- Inline documentation for device options and constraints
- No function definitions - pure configuration data
