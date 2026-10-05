import pytest
import torch

import inseq

from ._api import GradELLMCore


def test_registration():
    assert "grad_ellm" in inseq.list_feature_attribution_methods()


@pytest.mark.parametrize("method", ["grad_ellm", "deeplift"])
def test_decoder_only_subwords_then_explicit_scores(model, tokenizer, method):
    wrapped = inseq.load_model(model, method, tokenizer=tokenizer, device="cpu")
    out = wrapped.attribute(
        "the movie", generated_texts="the movie good positive", attributed_fn="logit", show_progress=False
    )
    # Reproduce the exact two calls in the user's screenshots.
    words = out.aggregate("subwords", aggregate_target=False)
    assert words[0].source_attributions is None
    assert words[0].target_attributions is not None
    scores = words.aggregate("scores", aggregate_target=True)
    default = words.aggregate()
    torch.testing.assert_close(scores[0].target_attributions, default[0].target_attributions, equal_nan=True)
    assert scores.show(do_aggregation=False, display=False, return_html=True)


def test_prompt_subword_spans_keep_generated_tokens():
    from inseq.data.aggregator import SubwordAggregator
    from inseq.data.grad_ellm import GradELLMSequenceOutput
    from inseq.utils.typing import TokenWithId

    tokens = [TokenWithId(t, i) for i, t in enumerate(["Ġmov", "ie", "Ġis", "Ġgood", "Ġpos", "itive"])]
    raw = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [float("nan"), 10.0, 11.0],
            [float("nan"), float("nan"), 12.0],
            [float("nan"), float("nan"), float("nan")],
        ]
    )
    seq = GradELLMSequenceOutput(
        source=tokens[:3],
        target=tokens,
        target_attributions=raw.clone(),
        attr_pos_start=3,
        step_scores={},
        sequence_scores={},
    )
    spans = SubwordAggregator.get_spans(seq.target[: seq.attr_pos_start], ("Ġ", "Ċ"), False)
    words = seq.aggregate("spans", source_spans=spans, target_spans=spans)
    assert words.attr_pos_start == 2
    assert words.target[words.attr_pos_start :] == tokens[3:]
    # Inseq's standard span function is absmax: merge mov + ie without changing output columns.
    expected = raw[[1, 2, 3, 4, 5]]
    torch.testing.assert_close(words.target_attributions, expected, equal_nan=True)
    scores = words.aggregate("scores", normalize=False)
    torch.testing.assert_close(scores.target_attributions, expected, equal_nan=True)
    assert scores.target_attributions[: scores.attr_pos_start].shape == (2, 3)
    assert words.aggregate("scores").show(display=False, return_html=True)


def test_teacher_forced_output_matches_core(model, tokenizer):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, n_layers=2, device="cpu")
    out = wrapped.attribute(
        "the movie is good",
        generated_texts="the movie is good positive negative",
        attributed_fn="logit",
        show_progress=False,
        output_step_attributions=True,
        step_scores=["probability"],
    )
    seq = out.sequence_attributions[0]
    ids = torch.tensor([[tok.id for tok in seq.target]])
    for j, step in enumerate(out.step_attributions):
        pos = seq.attr_pos_start + j
        expected = GradELLMCore(model, 2).attribute(ids[:, :pos], ids[:, pos])
        torch.testing.assert_close(step.target_attributions, expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(seq.target_attributions[:pos, j], expected[0], rtol=1e-6, atol=1e-8)
    assert seq.source_attributions is None
    assert seq.target_attributions.ndim == 2
    assert "probability" in seq.step_scores
    # Inseq fills future-token cells with NaN by design.
    assert torch.isnan(seq.target_attributions[-1]).all()
    assert wrapped.attribution_method.core._captures == {}
    # Visualize without implicit score normalization; already token-level scores.
    assert seq.show(do_aggregation=False, display=False, return_html=True) is not None


def test_real_generation(model, tokenizer):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    out = wrapped.attribute(
        "the movie",
        attributed_fn="logit",
        show_progress=False,
        generation_args={"max_new_tokens": 2, "do_sample": False},
    )
    assert out.info["attribution_method"] == "grad_ellm"
    assert out.sequence_attributions[0].target_attributions.ndim == 2


@pytest.mark.parametrize("generated_texts", ["the movie good", "the movie good positive negative"])
def test_default_show_normalizes_token_scores(model, tokenizer, generated_texts):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    out = wrapped.attribute("the movie", generated_texts=generated_texts, attributed_fn="logit", show_progress=False)
    raw = out.sequence_attributions[0].target_attributions.clone()
    # Exercise default aggregation through both the sequence and container APIs.
    assert out.show(display=False, return_html=True)
    assert out.sequence_attributions[0].show(display=False, return_html=True)
    aggregated = out.aggregate().sequence_attributions[0].target_attributions
    assert aggregated.shape == raw.shape
    # Compare against the same pipeline used by DeepLIFT and other gradient methods.
    from inseq.data.attribution import GranularFeatureAttributionSequenceOutput

    seq = out.sequence_attributions[0]
    gradient_output = GranularFeatureAttributionSequenceOutput(
        source=seq.source,
        target=seq.target,
        target_attributions=raw.unsqueeze(-1),
        attr_pos_start=seq.attr_pos_start,
        attr_pos_end=seq.attr_pos_end,
        step_scores={},
        sequence_scores={},
    )
    torch.testing.assert_close(aggregated, gradient_output.aggregate().target_attributions, equal_nan=True)
    torch.testing.assert_close(torch.nansum(aggregated, dim=0), torch.ones(aggregated.shape[1]))
    torch.testing.assert_close(out.aggregate(normalize=False)[0].target_attributions, raw, equal_nan=True)
    torch.testing.assert_close(out.sequence_attributions[0].target_attributions, raw, equal_nan=True)


def test_normalization_zero_columns_nan_and_repeated_aggregation():
    from inseq.data.grad_ellm import GradELLMSequenceOutput
    from inseq.utils.typing import TokenWithId

    tokens = [TokenWithId(str(i), i) for i in range(4)]
    raw = torch.tensor([[0.0, 2.0], [0.0, 4.0], [float("nan"), 2.0], [float("nan"), float("nan")]])
    seq = GradELLMSequenceOutput(
        source=tokens[:2],
        target=tokens,
        target_attributions=raw.clone(),
        attr_pos_start=2,
        step_scores={},
        sequence_scores={},
    )
    normalized = seq.aggregate()
    expected = torch.tensor([[0.0, 0.25], [0.0, 0.5], [float("nan"), 0.25], [float("nan"), float("nan")]])
    torch.testing.assert_close(normalized.target_attributions, expected, equal_nan=True)
    torch.testing.assert_close(normalized.aggregate().target_attributions, expected, equal_nan=True)
    torch.testing.assert_close(seq.target_attributions, raw, equal_nan=True)
    torch.testing.assert_close(seq.aggregate(normalize=False).target_attributions, raw, equal_nan=True)
    assert seq.show(display=False, return_html=True)


def test_normalization_rescale_option():
    import inspect

    from inseq.data.aggregator import SequenceAttributionAggregator
    from inseq.data.grad_ellm import GradELLMSequenceOutput
    from inseq.utils.typing import TokenWithId

    if "rescale" not in inspect.signature(SequenceAttributionAggregator._process_attribution_scores).parameters:
        pytest.skip("Inseq 0.6.0 has no rescale option; default L1 normalization is tested separately.")

    tokens = [TokenWithId(str(i), i) for i in range(3)]
    raw = torch.tensor([[2.0], [4.0], [float("nan")]])
    seq = GradELLMSequenceOutput(
        source=tokens[:2],
        target=tokens,
        target_attributions=raw.clone(),
        attr_pos_start=2,
        step_scores={},
        sequence_scores={},
    )
    scaled = seq.aggregate(normalize=False, rescale=True).target_attributions
    torch.testing.assert_close(scaled, torch.tensor([[0.5], [1.0], [float("nan")]]), equal_nan=True)


def test_normalized_output_save_load(model, tokenizer, tmp_path):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    out = wrapped.attribute(
        "the movie", generated_texts="the movie good positive", attributed_fn="logit", show_progress=False
    )
    path = str(tmp_path / "attributions.json")
    out.save(path)
    restored = inseq.FeatureAttributionOutput.load(path)
    torch.testing.assert_close(
        restored.aggregate()[0].target_attributions, out.aggregate()[0].target_attributions, equal_nan=True
    )
    assert restored.show(display=False, return_html=True)


@pytest.mark.parametrize("normalize", [True, False])
def test_inseq_batch_variable_lengths(model, tokenizer, normalize):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, n_layers=2, normalize=normalize, device="cpu")
    sources = ["the movie is good", "movie"]
    targets = ["the movie is good positive negative", "movie negative"]
    batched = wrapped.attribute(
        sources, generated_texts=targets, attributed_fn="logit", show_progress=False, batch_size=2
    )
    for source, target, batch_seq in zip(sources, targets, batched.sequence_attributions, strict=False):
        single = wrapped.attribute(source, generated_texts=target, attributed_fn="logit", show_progress=False)
        torch.testing.assert_close(
            batch_seq.target_attributions,
            single.sequence_attributions[0].target_attributions,
            equal_nan=True,
            rtol=2e-5,
            atol=2e-7,
        )


def test_probability_objective(model, tokenizer):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu", normalize=False)
    options = {"generated_texts": "the movie good", "show_progress": False, "output_step_attributions": True}
    out = wrapped.attribute("the movie", attributed_fn="probability", **options)
    score = out.step_attributions[0].target_attributions
    ids = torch.tensor([[4, 5]])
    expected = GradELLMCore(model, normalize=False).explain(
        lambda: model(ids, use_cache=False).logits[:, -1, :].softmax(-1)[:, 7], torch.ones_like(ids)
    )
    torch.testing.assert_close(score, expected, rtol=1e-6, atol=1e-8)


def test_switch_methods(model, tokenizer):
    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, device="cpu")
    for method in ["saliency", "grad_ellm", "deeplift", "attention", "grad_ellm"]:
        out = wrapped.attribute(
            "the movie", generated_texts="the movie good", method=method, attributed_fn="logit", show_progress=False
        )
        assert out.sequence_attributions[0].target_attributions is not None
        normalized = out.aggregate()[0].target_attributions
        assert normalized.ndim == 2
        torch.testing.assert_close(torch.nansum(normalized, dim=0), torch.ones(normalized.shape[1]))


def test_exact_ids_and_eos(model, tokenizer):
    from ._api import attribute_token_ids

    wrapped = inseq.load_model(model, "grad_ellm", tokenizer=tokenizer, n_layers=2, device="cpu")
    ids = torch.tensor([[1, 4, 5, 10, 2]])
    out = attribute_token_ids(wrapped, ids, prompt_length=3, show_progress=False)
    seq = out.sequence_attributions[0]
    assert [tok.id for tok in seq.target] == ids[0].tolist()
    assert seq.target_attributions.shape == (5, 2)
    for step, pos in enumerate([3, 4]):
        reference = GradELLMCore(model, 2).attribute(ids[:, :pos], ids[:, pos])
        torch.testing.assert_close(seq.target_attributions[:pos, step], reference[0])
