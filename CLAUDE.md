# CLAUDE.md - Inseq Project Documentation

## Project Overview

**Inseq** is a PyTorch-based interpretability toolkit for analyzing and explaining sequence generation models. It democratizes access to common post-hoc interpretability analyses for generative language models.

- **Version:** 0.7.0.dev0
- **Python Support:** 3.10, 3.11, 3.12, 3.13
- **License:** Apache License 2.0
- **Documentation:** https://inseq.org
- **Repository:** https://github.com/inseq-team/inseq

## Quick Start Commands

```bash
# Install package
make install

# Install with all development dependencies
make install-dev

# Run all tests
make test

# Run tests without GPU
make test-cpu

# Run fast tests only (skip slow tests)
make fast-test

# Check code style
make check-style

# Auto-format code
make fix-style

# Build documentation
make build-docs

# Serve docs locally
make serve-docs
```

## Directory Structure

```
inseq/
├── inseq/                    # Main package
│   ├── attr/                 # Attribution methods
│   │   ├── feat/             # Feature attribution implementations
│   │   │   └── ops/          # Custom operations (LIME, DIG, ReAGent, etc.)
│   │   └── step_functions.py # Custom step function registry
│   ├── models/               # Model loading and management
│   │   ├── attribution_model.py    # Base attribution model class
│   │   ├── encoder_decoder.py      # Encoder-decoder support
│   │   ├── decoder_only.py         # Decoder-only support
│   │   └── model_config.yaml       # Model configurations
│   ├── data/                 # Data structures
│   │   ├── attribution.py    # Attribution output classes
│   │   ├── aggregator.py     # Aggregation pipeline
│   │   ├── batch.py          # Batch data structures
│   │   └── viz.py            # Visualization utilities
│   ├── commands/             # CLI implementations
│   │   ├── attribute/        # Single example attribution
│   │   ├── attribute_dataset/# Dataset-wide attribution
│   │   └── attribute_context/# Context dependence detection
│   └── utils/                # Utility modules
├── tests/                    # Test suite (mirrors main package structure)
├── examples/                 # Example notebooks
├── docs/                     # Sphinx documentation
└── pyproject.toml            # Project configuration
```

## Key Modules

### Attribution Methods (`inseq/attr/`)

**Gradient-based:**
- `saliency`, `input_x_gradient`, `integrated_gradients`, `deeplift`, `gradient_shap`
- `discretized_integrated_gradients`, `sequential_integrated_gradients`

**Internals-based:**
- `attention` - Attention weight attribution

**Perturbation-based:**
- `occlusion`, `lime`, `value_zeroing`, `reagent`

### Models (`inseq/models/`)

- `AttributionModel` - Abstract base class
  - `DecoderOnlyAttributionModel` - GPT-2, LLaMA, etc.
  - `EncoderDecoderAttributionModel` - mBART, T5, etc.
- `load_model(model_id, attribution_method)` - Factory function

### Data Structures (`inseq/data/`)

- `FeatureAttributionOutput` - Top-level output container
- `FeatureAttributionSequenceOutput` - Single sequence results
- `Aggregator` / `AggregatorPipeline` - Post-processing chain
- `show_attributions()`, `show_granular_attributions()` - Visualization

### Step Functions (`inseq/attr/step_functions.py`)

Built-in scores: `logits`, `probability`, `entropy`, `crossentropy`, `perplexity`
Contrastive: `contrast_logits`, `contrast_prob`, `pcxmi`
Advanced: `kl_divergence`, `in_context_pvi`, `mc_dropout_prob_avg`

## Architecture Patterns

### Registry Pattern

All extensible components use registries for auto-discovery:

```python
class MyCustomAttribution(FeatureAttribution):
    method_name = "my_method"  # Auto-registered

# List all available
list_feature_attribution_methods()
```

### Input Formatter Protocol

Models implement `InputFormatter` for architecture-specific input handling:

```python
class InputFormatter(Protocol):
    @staticmethod
    def prepare_inputs_for_attribution(...) -> Batch
    @staticmethod
    def format_attribution_args(...) -> tuple
```

### Type System

Extensive type aliases in `utils/typing.py`:

```python
IdsTensor = Int["batch sequence_length"]
LogitsTensor = Float["batch sequence vocab"]
TextInput = str | list[str]
```

## Common API Usage

```python
import inseq

# Load model with attribution method
model = inseq.load_model("gpt2", "saliency")

# Run attribution
out = model.attribute(
    input_texts="The quick brown fox",
    generation_args={"max_new_tokens": 20},
    step_scores=["probability"]
)

# Visualize
out.show()

# Save/load
out.save("result.json", scores_precision="float16")
loaded = inseq.FeatureAttributionOutput.load("result.json")

# Aggregate
from inseq.data.aggregator import SubwordAggregator
aggregated = out.aggregate(SubwordAggregator)
```

## CLI Commands

```bash
# Single example attribution
inseq attribute --model gpt2 --method saliency --input "Hello world"

# Dataset attribution
inseq attribute-dataset --model gpt2 --method saliency --dataset my_dataset

# Context dependence
inseq attribute-context --model gpt2 --input "Context: ... Question: ..."
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=inseq tests/

# Skip slow tests
pytest -m "not slow" tests/

# Skip GPU tests
pytest -m "not require_cuda_gpu" tests/

# Run specific test file
pytest tests/attr/feat/test_feature_attribution.py
```

**Test Markers:**
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.require_cuda_gpu` - GPU-dependent tests

## Code Style

- **Formatter:** ruff (line length 119)
- **Type Checking:** mypy (strict mode, Python 3.10+)
- **Docstrings:** Google style with pydoclint
- **Pre-commit:** Configured in `.pre-commit-config.yaml`

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Auto-fix issues
ruff check --fix .
```

## Adding New Components

### New Attribution Method

```python
# inseq/attr/feat/my_method.py
from inseq.attr.feat import FeatureAttribution

class MyMethodAttribution(FeatureAttribution):
    method_name = "my_method"

    def attribute_step(self, ...):
        # Implementation
        pass
```

### New Step Function

```python
from inseq.attr.step_functions import register_step_function

@register_step_function
def my_score(args, attribution_model):
    # Return score tensor
    pass
```

### New Aggregator

```python
from inseq.data.aggregator import Aggregator

class MyAggregator(Aggregator):
    def aggregate(self, attr_output):
        # Implementation
        pass
```

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add attribution method | `inseq/attr/feat/`, `inseq/attr/feat/ops/` |
| Add step function | `inseq/attr/step_functions.py` |
| Add model support | `inseq/models/model_config.yaml` |
| Add aggregator | `inseq/data/aggregator.py` |
| Add CLI command | `inseq/commands/` |
| Add visualization | `inseq/data/viz.py` |

## Dependencies

**Core:** transformers, captum, torch, jaxtyping, matplotlib, rich, treescope, tqdm

**Optional:** datasets, ipywidgets, scikit-learn, nltk

## Serialization

Attribution outputs support JSON serialization with precision control:

```python
out.save("result.json")  # Default float32
out.save("result.json", scores_precision="float16")  # Reduced size
out.save("result.json", scores_precision="float8")   # Minimal size
```

## Caching

Two-tier caching in `~/.cache/inseq/`:

```python
from inseq.utils.cache import INSEQ_HOME_CACHE, INSEQ_ARTIFACTS_CACHE
```

## Error Handling

Custom exceptions in `inseq/utils/errors.py`:
- `UnknownAttributionMethodError`
- `MissingAttributionMethodError`
- `MissingAlignmentsError`
- `LengthMismatchError`

## Documentation

```bash
# Build docs
cd docs && make html

# Or use Makefile
make build-docs
make serve-docs
```

Documentation source is in `docs/source/` using Sphinx with Furo theme.
