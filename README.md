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
[![Demo Paper](https://img.shields.io/badge/ACL%20Anthology%20-%20?logo=data%3Aimage%2Fx-icon%3Bbase64%2CAAABAAEAIBIAAAEAIABwCQAAFgAAACgAAAAgAAAAJAAAAAEAIAAAAAAAAAkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACEa7k0mH%2B%2F5JBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJB3t%2FyQd7f8kHe3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQd7f8kHO3%2FJB3t%2FyQd7f8kHO3%2FJB3t%2FyQd7f8kHe3%2FJB3t%2FyMc79EkGP8VAAAAAAAAAAAAAAAAIRruTSYf7%2FkkHO3%2FJBzt%2FyQd7f8kHe3%2FJB3t%2FyQd7f8kHO3%2FJB3t%2FyQc7f8kHe3%2FJBzt%2FyQc7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FJB3t%2FyQd7f8kHe3%2FIxzv0SQY%2FxUAAAAAAAAAAAAAAAAhIe5NJh%2Fv%2BSQd7f8kHe3%2FJB3t%2FyQd7f8kHO3%2FJBzt%2FyQc7f8kHe3%2FJBzt%2FyQd7f8kHe3%2FJBzt%2FyQc7f8kHe3%2FJB3t%2FyQd7f8kHe3%2FJB3t%2FyQc7f8kHe3%2FJB3t%2FyQd7f8kHe3%2FJBzt%2FyQd7f8jHO%2FRJBj%2FFQAAAAAAAAAAAAAAACEa7k0mH%2B%2F5JB3t%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHe3%2FJBzt%2FyQd7f8kHe3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHe3%2FJBzt%2FyQd7f8kHe3%2FJB3t%2FyQc7f8kHe3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyMc79EkGP8VAAAAAAAAAAAAAAAAIRruTSYf7%2FkkHO3%2FJBzt%2FyQc7f8kHe3%2FJB3t%2FyQd7f8kHO3%2FJB3t%2FyQd7f8kHO3%2FJB3t%2FyQc7f8kHe3%2FJBzt%2FyQc7f8kHe3%2FJB3t%2FyQd7f8kHe3%2FJBzt%2FyQd7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FIxzv0SQY%2FxUAAAAAAAAAAAAAAAAhGu5NJh%2Fv%2BSQd7f8kHO3%2FJBzt%2FyQd7f8jIOzYIxvtgiQc8X8kHPF%2FJBzxfyQc8X8kHPF%2FJBzxfyQc8X8kHPF%2FIx%2FuiiMf7OgkHe3%2FJBzt%2FyQc7f8kHO3%2FJhzs9CUg7JYkHPF%2FJBzxfyQc8X8iHfBoMzP%2FCgAAAAAAAAAAAAAAACEa7k0mHu%2F5JBzt%2FyQc7f8kHO3%2FJBzt%2FyQb7LEAAP8FAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkGP8VIxzv0SQc7f8kHe3%2FJB3t%2FyQc7f8jHOzpIhzuLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIRruTSYf7%2FkkHe3%2FJB3t%2FyQd7f8kHO3%2FJBvssQAA%2FwUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQY%2FxUjHO%2FRJB3t%2FyQc7f8kHO3%2FJBzt%2FyMb7OkiHO4tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhGu5NJh%2Fv%2BSQd7f8kHe3%2FJBzt%2FyQc7f8kHuyxAAD%2FBQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJBj%2FFSMc79EkHe3%2FJBzt%2FyQc7f8kHO3%2FIxvs6SIc7i0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACEa7k0mH%2B%2F5JBzt%2FyQc7f8kHO3%2FJBzt%2FyQb7LEAAP8FAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkGP8VIx3v0SQd7f8kHe3%2FJBzt%2FyQc7f8jHOzpIhzuLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIRruTSYf7%2FkkHO3%2FJBzt%2FyQc7f8kHe3%2FJBzuxSgi81MhGu5NIRruTSEa7k0hGu5NISHuTSEh7k0hGu5NIRruTSIa72EjHe3aJBzt%2FyQd7f8kHO3%2FJBzt%2FyMc7OkiHO4tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhGu5NJh7v%2BSQc7f8kHe3%2FJBzt%2FyQc7f8kHO3%2FJh%2Fv%2BSYf7%2FkmHu%2F5Jh%2Fv%2BSYf7%2FkmH%2B%2F5Jh%2Fv%2BSYf7%2FkmH%2B%2F5Jh7v%2BSQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FIxzs6SIc7i0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACEa7k0mHu%2F5JBzt%2FyQd7f8kHO3%2FJBzt%2FyQc7f8kHe3%2FJB3t%2FyQd7f8kHO3%2FJBzt%2FyQc7f8kHe3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHe3%2FJBzt%2FyQd7f8jHOzpIhzuLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIRruTSYe7%2FkkHO3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyQd7f8kHe3%2FJB3t%2FyQc7f8kHe3%2FJB3t%2FyQd7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyMc7OkiHO4tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhGu5NJh%2Fv%2BSQc7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FIxzs6SIc7i0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACEa7k0mH%2B%2F5JB3t%2FyQc7f8kHe3%2FJBzt%2FyQc7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyQd7f8kHO3%2FJBzt%2FyQc7f8kHO3%2FJB3t%2FyQc7f8kHO3%2FJBzt%2FyQc7f8jHOzpIhzuLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKB%2FtOSUc7askHuyxJBvssSQe7LEkHuyxJB7ssSQe7LEkHuyxJBvssSQb7LEkHuyxJB7ssSQe7LEkHuyxJB7ssSUc7LMjHe31JB3t%2FyQd7f8kHe3%2FJBzt%2FyMc7OkiHO4tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD%2FBQAA%2FwUAAP8FAAD%2FBQAA%2FwUAAP8FAAD%2FBQAA%2FwUAAP8FAAD%2FBQAA%2FwUAAP8FAAD%2FBQAA%2FwUAAP8FHBzsGyMd7qYjHO%2FRIxzv0SMd79EjHO%2FRIx7tux4Y%2BSoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAABwAAAAcAAAAHAAAABwAAAAcAAAAHAP8A%2FwD%2FAP8A%2FwD%2FAP8A%2FwAAAP8AAAD%2FAAAA%2FwAAAP8AAAD%2FAAAA%2FwAAAP%2BAAAD8%3D&labelColor=white&color=red&link=https%3A%2F%2Faclanthology.org%2F2023.acl-demo.40%2F
)](http://arxiv.org/abs/2302.13942)

</div>
<div align="center">

  [![Follow Inseq on Twitter]( 	https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/InseqLib)
  [![Join the Inseq Discord server](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/V5VgwwFPbu)
  [![Read the Docs](https://img.shields.io/badge/-Docs-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://inseq.org)](https://inseq.org)
  [![Tutorial](https://img.shields.io/badge/-Tutorial-orange?style=for-the-badge&logo=Jupyter&logoColor=white&link=https://github.com/inseq-team/inseq/blob/main/examples/inseq_tutorial.ipynb)](https://github.com/inseq-team/inseq/blob/main/examples/inseq_tutorial.ipynb)


</div>

Inseq is a Pytorch-based hackable toolkit to democratize the access to common post-hoc **in**terpretability analyses of **seq**uence generation models.

## Installation

Inseq is available on PyPI and can be installed with `pip` for Python >= 3.9, <= 3.11:

```bash
# Install latest stable version
pip install inseq

# Alternatively, install latest development version
pip install git+https://github.com/inseq-team/inseq.git
```

Install extras for visualization in Jupyter Notebooks and 🤗 datasets attribution as `pip install inseq[notebook,datasets]`.

<details>
  <summary>Dev Installation</summary>
To install the package, clone the repository and run the following commands:

```bash
cd inseq
make uv-download # Download and install the UV package manager
make install # Installs the package and all dependencies
```

For library developers, you can use the `make install-dev` command to install all development dependencies (quality, docs, extras).

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

## Features

- 🚀 Feature attribution of sequence generation for most `ForConditionalGeneration` (encoder-decoder) and `ForCausalLM` (decoder-only) models from 🤗 Transformers

- 🚀 Support for multiple feature attribution methods, extending the ones supported by [Captum](https://captum.ai/docs/introduction)

- 🚀 Post-processing, filtering and merging of attribution maps via `Aggregator` classes.

- 🚀 Attribution visualization in notebooks, browser and command line.

- 🚀 Efficient attribution of single examples or entire 🤗 datasets with the Inseq CLI.

- 🚀 Custom attribution of target functions, supporting advanced methods such as [contrastive feature attributions](https://aclanthology.org/2022.emnlp-main.14/) and [context reliance detection](https://arxiv.org/abs/2310.01188).

- 🚀 Extraction and visualization of custom scores (e.g. probability, entropy) at every generation step alongsides attribution maps.

### Supported methods

Use the `inseq.list_feature_attribution_methods` function to list all available method identifiers and `inseq.list_step_functions` to list all available step functions. The following methods are currently supported:

#### Gradient-based attribution

- `saliency`: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (Simonyan et al., 2013)

- `input_x_gradient`: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (Simonyan et al., 2013)

- `integrated_gradients`: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017)

- `deeplift`: [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685) (Shrikumar et al., 2017)

- `gradient_shap`: [A unified approach to interpreting model predictions](https://dl.acm.org/doi/10.5555/3295222.3295230) (Lundberg and Lee, 2017)

- `discretized_integrated_gradients`: [Discretized Integrated Gradients for Explaining Language Models](https://aclanthology.org/2021.emnlp-main.805/) (Sanyal and Ren, 2021)

- `sequential_integrated_gradients`: [Sequential Integrated Gradients: a simple but effective method for explaining language models](https://aclanthology.org/2023.findings-acl.477/) (Enguehard, 2023)

#### Internals-based attribution

- `attention`: Attention Weight Attribution, from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)

#### Perturbation-based attribution

- `occlusion`: [Visualizing and Understanding Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) (Zeiler and Fergus, 2014)

- `lime`: ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) (Ribeiro et al., 2016)

#### Step functions

Step functions are used to extract custom scores from the model at each step of the attribution process with the `step_scores` argument in `model.attribute`. They can also be used as targets for attribution methods relying on model outputs (e.g. gradient-based methods) by passing them as the `attributed_fn` argument. The following step functions are currently supported:

- `logits`: Logits of the target token.
- `probability`: Probability of the target token. Can also be used for log-probability by passing `logprob=True`.
- `entropy`: Entropy of the predictive distribution.
- `crossentropy`: Cross-entropy loss between target token and predicted distribution.
- `perplexity`: Perplexity of the target token.
- `contrast_logits`/`contrast_prob`: Logits/probabilities of the target token when different contrastive inputs are provided to the model. Equivalent to `logits`/`probability` when no contrastive inputs are provided.
- `contrast_logits_diff`/`contrast_prob_diff`: Difference in logits/probability between original and foil target tokens pair, can be used for contrastive evaluation as in [contrastive attribution](https://aclanthology.org/2022.emnlp-main.14/) (Yin and Neubig, 2022).
- `pcxmi`: Point-wise Contextual Cross-Mutual Information (P-CXMI) for the target token given original and contrastive contexts [(Yin et al. 2021)](https://arxiv.org/abs/2109.07446).
- `kl_divergence`: KL divergence of the predictive distribution given original and contrastive contexts. Can be restricted to most likely target token options using the `top_k` and `top_p` parameters.
- `in_context_pvi`: In-context Pointwise V-usable Information (PVI) to measure the amount of contextual information used in model predictions [(Lu et al. 2023)](https://arxiv.org/abs/2310.12300).
- `mc_dropout_prob_avg`: Average probability of the target token across multiple samples using [MC Dropout](https://arxiv.org/abs/1506.02142) (Gal and Ghahramani, 2016).
- `top_p_size`: The number of tokens with cumulative probability greater than `top_p` in the predictive distribution of the model.

The following example computes contrastive attributions using the `contrast_prob_diff` step function:

```python
import inseq

attribution_model = inseq.load_model("gpt2", "input_x_gradient")

# Perform the contrastive attribution:
# Regular (forced) target -> "The manager went home because he was sick"
# Contrastive target      -> "The manager went home because she was sick"
out = attribution_model.attribute(
    "The manager went home because",
    "The manager went home because he was sick",
    attributed_fn="contrast_prob_diff",
    contrast_targets="The manager went home because she was sick",
    # We also visualize the corresponding step score
    step_scores=["contrast_prob_diff"]
)
out.show()
```

Refer to the [documentation](https://inseq.readthedocs.io/examples/custom_attribute_target.html) for an example including custom function registration.

## Using the Inseq CLI

The Inseq library also provides useful client commands to enable repeated attribution of individual examples and even entire 🤗 datasets directly from the console. See the available options by typing `inseq -h` in the terminal after installing the package.

Three commands are supported:

- `inseq attribute`: Wrapper for enabling `model.attribute` usage in console.

- `inseq attribute-dataset`: Extends `attribute` to full dataset using Hugging Face `datasets.load_dataset` API.

- `inseq attribute-context`: Detects and attribute context dependence for generation tasks using the approach of [Sarti et al. (2023)](https://arxiv.org/abs/2310.01188).

All commands support the full range of parameters available for `attribute`, attribution visualization in the console and saving outputs to disk.

<details>
  <summary><code>inseq attribute</code> example</summary>

  The following example performs a simple feature attribution of an English sentence translated into Italian using a MarianNMT translation model from <code>transformers</code>. The final result is printed to the console.
  ```bash
  inseq attribute \
  --model_name_or_path Helsinki-NLP/opus-mt-en-it \
  --attribution_method saliency \
  --input_texts "Hello world this is Inseq\! Inseq is a very nice library to perform attribution analysis"
  ```

</details>

<details>
  <summary><code>inseq attribute-dataset</code> example</summary>

  The following code can be used to perform attribution (both source and target-side) of Italian translations for a dummy sample of 20 English sentences taken from the FLORES-101 parallel corpus, using a MarianNMT translation model from Hugging Face <code>transformers</code>. We save the visualizations in HTML format in the file <code>attributions.html</code>. See the <code>--help</code> flag for more options.

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
</details>

<details>
  <summary><code>inseq attribute-context</code> example</summary>

  The following example uses a GPT-2 model to generate a continuation of <code>input_current_text</code>, and uses the additional context provided by <code>input_context_text</code> to estimate its influence on the the generation. In this case, the output <code>"to the hospital. He said he was fine"</code> is produced, and the generation of token <code>hospital</code> is found to be dependent on context token <code>sick</code> according to the <code>contrast_prob_diff</code> step function.

  ```bash
  inseq attribute-context \
    --model_name_or_path gpt2 \
    --input_context_text "George was sick yesterday." \
    --input_current_text "His colleagues asked him to come" \
    --attributed_fn "contrast_prob_diff"
  ```

  **Result:**

  ```
  Context with [contextual cues] (std λ=1.00) followed by output sentence with {context-sensitive target spans} (std λ=1.00)
  (CTI = "kl_divergence", CCI = "saliency" w/ "contrast_prob_diff" target)

  Input context:  George was sick yesterday.
  Input current: His colleagues asked him to come
  Output current: to the hospital. He said he was fine

  #1.
  Generated output (CTI > 0.428): to the {hospital}(0.548). He said he was fine
  Input context (CCI > 0.460):    George was [sick](0.516) yesterday.
  ```
</details>

## Planned Development

- ⚙️ Support more attention-based and occlusion-based feature attribution methods (documented in [#107](https://github.com/inseq-team/inseq/issues/107) and [#108](https://github.com/inseq-team/inseq/issues/108)).

- ⚙️ Interoperability with [ferret](https://ferret.readthedocs.io/en/latest/) for attribution plausibility and faithfulness evaluation.

- ⚙️ Rich and interactive visualizations in a tabbed interface using [Gradio Blocks](https://gradio.app/docs/#blocks).

## Contributing

Our vision for Inseq is to create a centralized, comprehensive and robust set of tools to enable fair and reproducible comparisons in the study of sequence generation models. To achieve this goal, contributions from researchers and developers interested in these topics are more than welcome. Please see our [contributing guidelines](CONTRIBUTING.md) and our [code of conduct](CODE_OF_CONDUCT.md) for more information.

## Citing Inseq

If you use Inseq in your research we suggest to include a mention to the specific release (e.g. v0.4.0) and we kindly ask you to cite our reference paper as:

```bibtex
@inproceedings{sarti-etal-2023-inseq,
    title = "Inseq: An Interpretability Toolkit for Sequence Generation Models",
    author = "Sarti, Gabriele  and
      Feldhus, Nils  and
      Sickert, Ludwig  and
      van der Wal, Oskar and
      Nissim, Malvina and
      Bisazza, Arianna",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-demo.40",
    doi = "10.18653/v1/2023.acl-demo.40",
    pages = "421--435",
}

```

## Research using Inseq

Inseq has been used in various research projects. A list of known publications that use Inseq to conduct interpretability analyses of generative models is shown below. If you know more, please let us know or submit a pull request (*last updated: February 2024*).

<details>
  <summary><b>2023</b></summary>
  <ol>
    <li> <a href="https://aclanthology.org/2023.acl-demo.40/">Inseq: An Interpretability Toolkit for Sequence Generation Models</a> (Sarti et al., 2023) </li>
    <li> <a href="https://arxiv.org/abs/2302.14220">Are Character-level Translations Worth the Wait? Comparing ByT5 and mT5 for Machine Translation</a> (Edman et al., 2023) </li>
    <li> <a href="https://aclanthology.org/2023.nlp4convai-1.1/">Response Generation in Longitudinal Dialogues: Which Knowledge Representation Helps?</a> (Mousavi et al., 2023)  </li>
    <li> <a href="https://openreview.net/forum?id=XTHfNGI3zT">Quantifying the Plausibility of Context Reliance in Neural Machine Translation</a> (Sarti et al., 2023)</li>
    <li> <a href="https://aclanthology.org/2023.emnlp-main.243/">A Tale of Pronouns: Interpretability Informs Gender Bias Mitigation for Fairer Instruction-Tuned Machine Translation</a> (Attanasio et al., 2023)</li>
    <li> <a href="https://arxiv.org/abs/2310.09820">Assessing the Reliability of Large Language Model Knowledge</a> (Wang et al., 2023)</li>
    <li> <a href="https://aclanthology.org/2023.conll-1.18/">Attribution and Alignment: Effects of Local Context Repetition on Utterance Production and Comprehension in Dialogue</a> (Molnar et al., 2023)</li>
  </ol>

</details>

<details>
  <summary><b>2024</b></summary>
  <ol>
    <li><a href="https://arxiv.org/abs/2401.12576">LLMCheckup: Conversational Examination of Large Language Models via Interpretability Tools</a> (Wang et al., 2024)</li>
    <li><a href="https://arxiv.org/abs/2402.00794">ReAGent: A Model-agnostic Feature Attribution Method for Generative Language Models</a> (Zhao et al., 2024)</li>
  </ol>

</details>
