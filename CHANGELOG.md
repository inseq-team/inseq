# Changelog

*This file contains a high-level description of changes that were merged into the Inseq main branch since the last release. Refer to the [releases page](https://github.com/inseq-team/inseq/releases) for an exhaustive overview of changes introduced at each release.*

## 🚀 Features

- Support for multi-GPU attribution ([#238](https://github.com/inseq-team/inseq/pull/238))
- Added `inseq attribute-context` CLI command to support the [PECoRe framework] for detecting and attributing context reliance in generative LMs ([#237](https://github.com/inseq-team/inseq/pull/237))

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

## 📝 Documentation and Tutorials

*No changes*

## 💥 Breaking Changes

*No changes*
