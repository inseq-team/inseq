<div align="center">
  <h1>amseq</h1>
  <h3>Attribution methods for sequence-to-sequence transformer models 🔍</h3>
</div>

<div align="center">

[![Build status](https://github.com/gsarti/amseq/workflows/build/badge.svg?branch=master&event=push)](https://github.com/gsarti/amseq/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/amseq.svg)](https://pypi.org/project/amseq/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/gsarti/amseq/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/gsarti/amseq/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/gsarti/amseq/releases)
[![License](https://img.shields.io/github/license/gsarti/amseq)](https://github.com/gsarti/amseq/blob/master/LICENSE)



</div>

## Example usage

```python
import logging
from amseq import AttributionModel, GradientAttributionOutput, heatmap

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
model_name = f'Helsinki-NLP/opus-mt-en-it'
model = AttributionModel(model_name)
model = amseq.load("Helsinki-NLP/opus-mt-en-it", "integrated_gradients")
sample_texts = ["Hello world, today is a good day!"]
out = model.attribute(txt, references=None, attr_pos_end=None, return_convergence_delta=True, n_steps=300)
```

![En-It Attribution Heatmap](img/heatmap_helloworld_enit.png)

## Todos

- [x] Generalize to other HF models
- [x] Generalize to other attribution methods
- [x] Add constrained attribution
- [x] Allow for batched attribution
- Integrate fairseq models

## Feature attribution steps

1. **Define model name or path and attribution method**

```python
model_name = f"Helsinki-NLP/opus-mt-en-it"
method = "integrated_gradients"
```

2. **Initialize model**

```python
model = AttributionModel.load(model_name)
```

What happens under the hood:

- Disambiguate whether it's a HF or a fairseq model and load it and the tokenizer.

- If the method is passed to the attribution model, perform the setup

3. **Attribute a sample text**

```python
text = "The ultimate test of your knowledge is your capacity to convey it to another."
out = model.attribute(text, method=method)
```

What happens under the hood:

- Check if method parameter is defined and if it matches the one at initialization. If not, replace it for this time only. If missing, use the one at initialization. If both missing, raise an error. If not currently in use, perform the setup for the attribution method.

- Check if a reference text is defined. If not, we perform a standard greedy decoding of the target (pick the highest value in the logits), else we tokenize the reference text and force the model to compute the attributions and decode the reference by picking the corresponding value in the logits.

- Return one or more FeatureAttributionOutput objects.
