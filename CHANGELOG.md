# Changelog

*This file contains a high-level description of changes that were merged into the Inseq main branch since the last release. Refer to the [releases page](https://github.com/inseq-team/inseq/releases) for an exhaustive overview of changes introduced at each release.*

## 🚀 Features

- Added new models `DbrxForCausalLM`, `OlmoForCausalLM`, `Phi3ForCausalLM`, `Qwen2MoeForCausalLM` to model config.

## 🔧 Fixes and Refactoring

- Fix the issue in the attention implementation from [#268](https://github.com/inseq-team/inseq/issues/268) where non-terminal position in the tensor were set to nan if they were 0s ([#269](https://github.com/inseq-team/inseq/pull/269)).

- Fix the pad token in cases where it is not specified by default in the loaded model (e.g. for Qwen models) ([#269](https://github.com/inseq-team/inseq/pull/269)).

- Fix bug reported in [#266](https://github.com/inseq-team/inseq/issues/266) making `value_zeroing` unusable for SDPA attention. This enables using the method on models using SDPA attention as default (e.g. `GemmaForCausalLM`) without passing `model_kwargs={'attn_implementation': 'eager'}` ([#267](https://github.com/inseq-team/inseq/pull/267)).

- Fix multi-device support and duplicate BOS for chat template models ([#280](https://github.com/inseq-team/inseq/pull/280)).

- Add `rescale_attributions` to Inseq CLI commands for `rescale=True` ([#280](https://github.com/inseq-team/inseq/pull/280)).

## 📝 Documentation and Tutorials

*No changes*

## 💥 Breaking Changes

*No changes*
