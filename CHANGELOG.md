# Changelog

*This file contains a high-level description of changes that were merged into the Inseq main branch since the last release. Refer to the [releases page](https://github.com/inseq-team/inseq/releases) for an exhaustive overview of changes introduced at each release.*

## 🚀 Features

- Support for multi-GPU attribution ([#238](https://github.com/inseq-team/inseq/pull/238))
- Added `inseq attribute-context` CLI command to support the [PECoRe framework] for detecting and attributing context reliance in generative LMs ([#237](https://github.com/inseq-team/inseq/pull/237))
- Added `value_zeroing` (`inseq.attr.feat.perturbation_attribution.ValueZeroingAttribution`) attribution method ([#173](https://github.com/inseq-team/inseq/pull/173))
- Added `rollout` (`inseq.data.aggregation_functions.RolloutAggregationFunction`) aggregation function for `SequenceAttributionAggregator` class ([#173](https://github.com/inseq-team/inseq/pull/173)).
- `value_zeroing` and `attention` use scores from the last generation step to produce outputs more efficiently (`is_final_step_method = True`) ([#173](https://github.com/inseq-team/inseq/pull/173)).

## 🔧 Fixes & Refactoring

- Fix `ContiguousSpanAggregator` and `SubwordAggregator` edge case of single-step generation ([#247](https://github.com/inseq-team/inseq/pull/247))
- Move tensors to CPU right away in the forward pass to avoid OOM when cloning ([#245](https://github.com/inseq-team/inseq/pull/245))
- Fix `remap_from_filtered` behavior on sequence_scores tensors. ([#245](https://github.com/inseq-team/inseq/pull/245))
- Use torch-native padding when converting lists of `FeatureAttributionStepOutput` to `FeatureAttributionSequenceOutput` in `get_sequences_from_batched_steps`. ([#245](https://github.com/inseq-team/inseq/pull/245))
- Bump `ruff` version ([#245](https://github.com/inseq-team/inseq/pull/245))
- Drop `poetry` in favor of [`uv`](https://github.com/astral-sh/uv) to accelerate package installation and simplify config in `pyproject.toml`. ([#249](https://github.com/inseq-team/inseq/pull/249))
- Drop `darglint` in favor of `pydoclint`. ([#249](https://github.com/inseq-team/inseq/pull/249))
- Replace Arxiv with ACL Anthology badge in `README`. ([#249](https://github.com/inseq-team/inseq/pull/249))
- Add first version of `CHANGELOG.md` ([#249](https://github.com/inseq-team/inseq/pull/249))
- Added multithread support for running tests using `pytest-xdist`

## 📝 Documentation and Tutorials

*No changes*

## 💥 Breaking Changes

- If `attention` is used as attribution method in `model.attribute`, `step_scores` cannot be extracted at the same time since the method does not require iterating over the full sequence anymore. ([#173](https://github.com/inseq-team/inseq/pull/173)) As an alternative, step scores can be extracted separately using the `dummy` attribution method (i.e. no attribution).
- BOS is always included in target-side attribution and generated sequences if present. ([#173](https://github.com/inseq-team/inseq/pull/173))
