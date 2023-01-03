<div align="center">
  <img src="https://raw.githubusercontent.com/inseq-team/inseq/main/docs/source/images/inseq_logo.png" width="300"/>
  <h4>Intepretability for Sequence Generation Models 🔍</h4>
</div>
<br/>
<div align="center">

[![Build status](https://img.shields.io/github/actions/workflow/status/inseq-team/inseq/build.yml?branch=main)](https://github.com/inseq-team/inseq/actions?query=workflow%3Abuild)
[![Docs status](https://img.shields.io/readthedocs/inseq)](https://inseq.readthedocs.io)
[![Version](https://img.shields.io/pypi/v/inseq?color=blue)](https://pypi.org/project/inseq/)
[![Python Version](https://img.shields.io/pypi/pyversions/inseq.svg?color=blue)](https://pypi.org/project/inseq/)
[![Downloads](https://static.pepy.tech/badge/inseq)](https://pepy.tech/project/inseq)
[![License](https://img.shields.io/github/license/inseq-team/inseq)](https://github.com/inseq-team/inseq/blob/main/LICENSE)

</div>
<div align="center">

  [![Follow Inseq on Twitter](https://img.shields.io/twitter/follow/inseqdev?label=Inseqdev&style=social)](https://twitter.com/InseqDev)
  [![Follow Inseq on Mastodon](https://img.shields.io/mastodon/follow/109308976376923913?domain=https%3A%2F%2Fsigmoid.social&label=Inseq&style=social)](https://sigmoid.social/@inseq)
</div>

Inseq is a Pytorch-based hackable toolkit to democratize the access to common post-hoc **in**terpretability analyses of **seq**uence generation models.

## Installation

Inseq is available on PyPI and can be installed with `pip`:

```bash
pip install inseq
```

Install extras for visualization in Jupyter Notebooks and 🤗 datasets attribution as `pip install inseq[notebook,datasets]`.

<details>
  <summary>Dev Installation</summary>
To install the package, clone the repository and run the following commands:

```bash
cd inseq
make poetry-download # Download and install the Poetry package manager
make install # Installs the package and all dependencies
```

If you have a GPU available, use `make install-gpu` to install the latest `torch` version with GPU support.

For library developers, you can use the `make install-dev` command to install and its GPU-friendly counterpart `make install-dev-gpu` to install all development dependencies (quality, docs, extras).

After installation, you should be able to run `make fast-test` and `make lint` without errors.
</details>

<details>
  <summary>FAQ Installation</summary>

  - Installing the `tokenizers` package requires a Rust compiler installation. You can install Rust from [https://rustup.rs](https://rustup.rs) and add `$HOME/.cargo/env` to your PATH.

  - Installing `sentencepiece` requires various packages, install with `sudo apt-get install cmake build-essential pkg-config` or `brew install cmake gperftools pkg-config`.
</details>

## Example usage in Python

This example uses the Integrated Gradients attribution method to attribute the English-French translation of a sentence taken from the WinoMT corpus:

```python
import inseq

model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "integrated_gradients")
out = model.attribute(
  "The developer argued with the designer because her idea cannot be implemented.",
  n_steps=100
)
out.show()
```

This produces a visualization of the attribution scores for each token in the input sentence (token-level aggregation is handled automatically). Here is what the visualization looks like inside a Jupyter Notebook:

![WinoMT Attribution Map](https://raw.githubusercontent.com/inseq-team/inseq/main/docs/source/images/heatmap_winomt.png)

Inseq also supports decoder-only models such as [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html), enabling usage of a variety of attribution methods and customizable settings directly from the console:

```python
import inseq

model = inseq.load_model("gpt2", "integrated_gradients")
model.attribute(
    "Hello ladies and",
    generation_args={"max_new_tokens": 9},
    n_steps=500,
    internal_batch_size=50
).show()
```

![GPT-2 Attribution in the console](https://raw.githubusercontent.com/inseq-team/inseq/main/docs/source/images/inseq_python_console.gif)

## Current Features

- 🚀 Feature attribution of sequence generation for most `ForConditionalGeneration` (encoder-decoder) and `ForCausalLM` (decoder-only) models from 🤗 Transformers

- 🚀 Support for single and batched attribution using multiple gradient-based feature attribution methods from [Captum](https://captum.ai/docs/introduction)

- 🚀 Post-hoc aggregation of feature attribution maps via `Aggregator` classes.

- 🚀 Attribution visualization in notebooks, browser and command line.

- 🚀 CLI for attributing single examples or entire 🤗 datasets.

- 🚀 Custom attribution of target functions, supporting advanced use cases such as contrastive and uncertainty-weighted feature attributions.

- 🚀 Extraction and visualization of custom step scores (e.g. probability, entropy) alongsides attribution maps.

## Planned Development

- ⚙️ Attention-based and occlusion-based feature attribution methods (documented in [#107](https://github.com/inseq-team/inseq/issues/107) and [#108](https://github.com/inseq-team/inseq/issues/108)).

- ⚙️ Interoperability with [ferret](https://ferret.readthedocs.io/en/latest/) for attribution evaluation.

- ⚙️ Rich and interactive visualizations in a tabbed interface using [Gradio Blocks](https://gradio.app/docs/#blocks).

- ⚙️ Baked-in advanced capabilities for contrastive and uncertainty-weighted feature attribution.

## Using the Inseq client

The Inseq library also provides useful client commands to enable repeated attribution of individual examples and even entire 🤗 datasets directly from the console. See the available options by typing `inseq -h` in the terminal after installing the package.

For now, two commands are supported:

- `ìnseq attribute`: Wraps the `attribute` method shown above, requires explicit inputs to be attributed.

- `inseq attribute-dataset`: Enables attribution for a full dataset using Hugging Face `datasets.load_dataset`.

Both commands support the full range of parameters available for `attribute`, attribution visualization in the console and saving outputs to disk.

**Example:** The following command can be used to perform attribution (both source and target-side) of Italian translations for a dummy sample of 20 English sentences taken from the FLORES-101 parallel corpus, using a MarianNMT translation model from Hugging Face `transformers`. We save the visualizations in HTML format in the file `attributions.html`. See the `--help` flag for more options.

```bash
inseq attribute-dataset \
  --model_name_or_path Helsinki-NLP/opus-mt-en-it \
  --attribution_method saliency \
  --do_prefix_attribution \
  --dataset_name inseq/dummy_enit \
  --input_text_field en \
  --dataset_split "train[:20]" \
  --viz_path attributions.html \
  --batch_size 8 \
  --hide
```

## Contributing

Our vision for Inseq is to create a centralized, comprehensive and robust set of tools to enable fair and reproducible comparisons in the study of sequence generation models. To achieve this goal, contributions from researchers and developers interested in these topics are more than welcome. Please see our [contributing guidelines](CONTRIBUTING.md) and our [code of conduct](CODE_OF_CONDUCT.md) for more information.
