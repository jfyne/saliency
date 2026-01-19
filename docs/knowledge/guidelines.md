# Development Guidelines

## Code Patterns to Prefer

### Modular Structure
- Maintain clear separation: `data.py`, `model.py`, `config.py`, `utils.py`
- Each module has single responsibility
- Keep `main.py` as thin CLI wrapper

### Configuration Management
- All hyperparameters and settings in `config.py`
- Use `PARAMS` dictionary for training configuration
- Device selection via `device` parameter (`"gpu"`, `"cpu"`, `"tpu"`, `"metal"`)

### TensorFlow/Keras Best Practices
- Follow existing patterns in `model.py` for layer definitions
- Use functional API for complex architectures
- Explicit data format handling (`channels_first` vs `channels_last`)
- Leverage tf.data pipeline patterns from `data.py`

### Explicit Over Implicit
- Explicit data format declarations
- Clear function signatures with type hints (where used)
- Document device-specific behavior in docstrings

## Patterns to Avoid

### Hardcoded Values
- ❌ Never hardcode paths, URLs, or hyperparameters
- ✅ Use `config.py` or function parameters
- ❌ Magic numbers in model architecture
- ✅ Named constants or config values

### Cross-Platform Compatibility Breaks
- ❌ Assuming specific data format without device check
- ✅ Use device-aware data format selection
- ❌ Platform-specific code without fallbacks
- ✅ Graceful degradation or clear error messages

### Undocumented Breaking Changes
- ❌ Changing data formats without updating docs/knowledge/weight-format-compatibility.md
- ✅ Document weight compatibility impact
- ❌ Modifying model architecture without version bump
- ✅ Track architecture changes for reproducibility

### Dependency Version Conflicts
- ❌ Adding dependencies without checking tensorflow-metal constraints
- ✅ Verify compatibility across all extras (gpu, metal, tpu)
- ❌ Upgrading TensorFlow beyond 2.18.1 without testing Metal
- ✅ Test or document platform-specific version constraints
