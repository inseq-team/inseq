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
sample_text = "Hello world, today is a good day!"
out: GradientAttributionOutput = model.attribute(sample_text, n_steps=10)
heatmap(out)
```

```shell
Original: "Hello world, today is a good day!"
Generating: ▁Ciao▁mondo,▁oggi▁è▁una▁buona▁giornata!: : 12it [00:16, 1.40s/it]
Generated: "Ciao mondo, oggi è una buona giornata!"
```

![En-It Attribution Heatmap](img/heatmap_enit.png)

## Todos

- Generalize to other models
- Generalize to other attribution methods
- Add constrained attribution
