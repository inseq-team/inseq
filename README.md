<div align="center">
  <img src="https://github.com/inseq-team/inseq/docs/source/images/inseq_logo.png" width="480"/>
  <h3>Intepretability for Sequence-to-sequence Models 🔍</h3>
</div>
<br/>
<div align="center">

[![Build status](https://github.com/inseq-team/inseq/workflows/build/badge.svg?branch=master&event=push)](https://github.com/inseq-team/inseq/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/inseq.svg)](https://pypi.org/project/inseq/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/inseq-team/inseq/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/inseq-team/inseq/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/inseq-team/inseq/releases)
[![License](https://img.shields.io/github/license/inseq-team/inseq)](https://github.com/inseq-team/inseq/blob/master/LICENSE)

</div>

## Example usage

This example uses the Integrated Gradients attribution method to attribute the English-French translation of a sentence taken from the WinoMT corpus:

```python
from inseq import load_model

model = inseq.load_model("Helsinki-NLP/opus-mt-en-it", "integrated_gradients")
out = model.attribute(
  "The developer argued with the designer because her idea cannot be implemented.",
  return_convergence_delta=True,
  n_steps=100
)
out.show()
```

![WinoMT Attribution Map](docs/source/images/heatmap_winomt.png)
