# Changelog

*This file contains a high-level description of changes that were merged into the Inseq main branch since the last release. Refer to the [releases page](https://github.com/inseq-team/inseq/releases) for an exhaustive overview of changes introduced at each release.*

## 🚀 Features

- Added [treescope](https://github.com/google-deepmind/treescope) for model and tensor visualization.

- Added new models `DbrxForCausalLM`, `OlmoForCausalLM`, `Phi3ForCausalLM`, `Qwen2MoeForCausalLM`, `Gemma2ForCausalLM` to model config.

- Add `rescale_attributions` to Inseq CLI commands for `rescale=True` ([#280](https://github.com/inseq-team/inseq/pull/280)).

- Rows and columns in the visualization now have indices alongside tokens to facilitate index-based slicing, aggregation and alignment [#282](https://github.com/inseq-team/inseq/pull/282)

- Added a `scores_precision` to `FeatureAttributionOutput.save` to enable efficient saving in `float16` and `float8` formats. This is useful for saving large attribution outputs in a more memory-efficient way. [#273](https://github.com/inseq-team/inseq/pull/273)

```python
import inseq

attrib_model = inseq.load_model("gpt2", "attention")
out = attrib_model.attribute("Hello world", generation_kwargs={'max_new_tokens': 100})

# Previous usage, memory inefficient
out.save("output.json")

# Memory-efficient saving
out.save("output_fp16.json", scores_precision="float16") # or "float8"

# Automatic conversion to float32
out_loaded = inseq.FeatureAttributionOutput.load("output_fp16.json")
```

- - A new `SliceAggregator` (`"slices"`) is added to allow for slicing source (in encoder-decoder) or target (in decoder-only) tokens from a `FeatureAttributionSequenceOutput` object, using the same syntax of `ContiguousSpanAggregator`. The `__getitem__` method of the `FeatureAttributionSequenceOutput` is a shortcut for this, allowing slicing with `[start:stop]` syntax. [#282](https://github.com/inseq-team/inseq/pull/282)

```python
import inseq
from inseq.data.aggregator import SliceAggregator

attrib_model = inseq.load_model("gpt2", "attention")
input_prompt = """Instruction: Summarize this article.
Input_text: In a quiet village nestled between rolling hills, an ancient tree whispered secrets to those who listened. One night, a curious child named Elara leaned close and heard tales of hidden treasures beneath the roots. As dawn broke, she unearthed a shimmering box, unlocking a forgotten world of wonder and magic.
Summary:"""

full_output_prompt = input_prompt + " Elara discovers a shimmering box under an ancient tree, unlocking a world of magic."

out = attrib_model.attribute(input_prompt, full_output_prompt)[0]

# These are all equivalent ways to slice only the input text contents
out_sliced = out.aggregate(SliceAggregator, target_spans=(13,73))
out_sliced = out.aggregate("slices", target_spans=(13,73))
out_sliced = out[13:73]
```

- The `__sub__` method in `FeatureAttributionSequenceOutput` is now used as a shortcut for `PairAggregator` [#282](https://github.com/inseq-team/inseq/pull/282)


```python
import inseq

attrib_model = inseq.load_model("gpt2", "saliency")

out_male = attrib_model.attribute(
    "The director went home because",
    "The director went home because he was tired",
    step_scores=["probability"]
)[0]
out_female = attrib_model.attribute(
    "The director went home because",
    "The director went home because she was tired",
    step_scores=["probability"]
)[0]
(out_male - out_female).show()
```

## 🔧 Fixes and Refactoring

- Fix the issue in the attention implementation from [#268](https://github.com/inseq-team/inseq/issues/268) where non-terminal position in the tensor were set to nan if they were 0s ([#269](https://github.com/inseq-team/inseq/pull/269)).

- Fix the pad token in cases where it is not specified by default in the loaded model (e.g. for Qwen models) ([#269](https://github.com/inseq-team/inseq/pull/269)).

- Fix bug reported in [#266](https://github.com/inseq-team/inseq/issues/266) making `value_zeroing` unusable for SDPA attention. This enables using the method on models using SDPA attention as default (e.g. `GemmaForCausalLM`) without passing `model_kwargs={'attn_implementation': 'eager'}` ([#267](https://github.com/inseq-team/inseq/pull/267)).

- Fix multi-device support and duplicate BOS for chat template models ([#280](https://github.com/inseq-team/inseq/pull/280)).

- The directions of generated/attributed tokens were clarified in the visualization using arrows instead of x/y [#282](https://github.com/inseq-team/inseq/pull/282)

## 📝 Documentation and Tutorials

*No changes*

## 💥 Breaking Changes

*No changes*
