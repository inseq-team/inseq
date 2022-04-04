<div align="center">
  <img src="/docs/source/images/inseq_logo.png" width="300"/>
  <h4>Intepretability for Sequence-to-sequence Models 🔍</h4>
</div>
<br/>
<div align="center">

[![Build status](https://github.com/inseq-team/inseq/workflows/build/badge.svg?branch=master&event=push)](https://github.com/inseq-team/inseq/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/inseq.svg)](https://pypi.org/project/inseq/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/inseq-team/inseq)](https://github.com/inseq-team/inseq/blob/main/LICENSE)

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
