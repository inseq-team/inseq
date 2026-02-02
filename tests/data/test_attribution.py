from unittest.mock import MagicMock

import torch
from pytest import fixture

from inseq import FeatureAttributionOutput, load_model
from inseq.data.attribution import FeatureAttributionSequenceOutput
from inseq.utils.typing import TokenWithId


@fixture(scope="session")
def saliency_mt_model():
    return load_model("Helsinki-NLP/opus-mt-en-it", "saliency")


@fixture(scope="session")
def saliency_gpt2_model_tiny():
    return load_model("hf-internal-testing/tiny-random-GPT2LMHeadModel", "saliency")


def test_save_load_attribution(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr.json"
    out = saliency_mt_model.attribute("This is a test.", device="cpu", show_progress=False)
    out.save(out_path)
    loaded_out = FeatureAttributionOutput.load(out_path)
    assert out == loaded_out


def test_save_load_attribution_split(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr.json"
    out = saliency_mt_model.attribute(["This is a test.", "sequence number two"], device="cpu", show_progress=False)
    out.save(out_path, split_sequences=True)
    out_path_1 = tmp_path / "tmp_attr_0.json"
    loaded_out = FeatureAttributionOutput.load(out_path_1)
    assert torch.allclose(
        out.sequence_attributions[0].source_attributions, loaded_out.sequence_attributions[0].source_attributions
    )


def test_save_load_attribution_compressed(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr_compress.json.gz"
    out = saliency_mt_model.attribute("This is a test.", device="cpu", show_progress=False)
    out.save(out_path, compress=True)
    loaded_out = FeatureAttributionOutput.load(out_path, decompress=True)
    assert out == loaded_out


def test_save_load_attribution_float16(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr_compress.json.gz"
    out = saliency_mt_model.attribute("This is a test.", device="cpu", show_progress=False)
    out.save(out_path, compress=True, scores_precision="float16")
    loaded_out = FeatureAttributionOutput.load(out_path, decompress=True)
    assert torch.allclose(
        out.sequence_attributions[0].source_attributions,
        loaded_out.sequence_attributions[0].source_attributions,
        atol=1e-05,
    )


def test_save_load_attribution_float8(tmp_path, saliency_mt_model):
    out_path = tmp_path / "tmp_attr_compress.json.gz"
    out = saliency_mt_model.attribute("This is a test.", device="cpu", show_progress=False)
    out.save(out_path, compress=True, scores_precision="float8")
    loaded_out = FeatureAttributionOutput.load(out_path, decompress=True)
    assert torch.allclose(
        out.sequence_attributions[0].source_attributions,
        loaded_out.sequence_attributions[0].source_attributions,
        atol=1e-02,
    )


def test_get_scores_dicts_encoder_decoder(saliency_mt_model):
    out = saliency_mt_model.attribute(["This is a test.", "Hello world!"], device="cpu", show_progress=False)
    dicts = out.get_scores_dicts()
    assert len(dicts) == 2
    assert isinstance(dicts[0], dict) and isinstance(dicts[1], dict)
    assert "source_attributions" in dicts[0] and "target_attributions" in dicts[0] and "step_scores" in dicts[0]


def test_get_scores_dicts_decoder_only(saliency_gpt2_model_tiny):
    out = saliency_gpt2_model_tiny.attribute(
        ["This is a test", "Hello world!"],
        ["This is a test generation", "Hello world! Today is a beautiful day."],
        show_progress=False,
        device="cpu",
    )
    dicts = out.get_scores_dicts()
    assert len(dicts) == 2
    assert isinstance(dicts[0], dict) and isinstance(dicts[1], dict)
    assert "source_attributions" in dicts[0] and "target_attributions" in dicts[0] and "step_scores" in dicts[0]


def test_decode_tokens_byte_level_tokenizer():
    """Test decode_tokens for byte-level tokenizers (issue #301).

    Byte-level tokenizers like Qwen return raw vocabulary entries from convert_ids_to_tokens
    that may be unreadable (garbled bytes for non-ASCII characters). The decode_tokens method
    should convert these to human-readable text using the tokenizer's decode method.
    """
    # Simulate byte-level tokenizer behavior:
    # - Raw tokens from vocab are garbled bytes (e.g., "\xe4\xbd\xa0\xe5\xa5\xbd" for "你好")
    # - decode() returns the proper human-readable text
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda ids: {
        100: "你好",  # Chinese "hello"
        101: "世界",  # Chinese "world"
        102: "！",  # Chinese exclamation mark
    }.get(ids[0], f"<unk:{ids[0]}>")

    # Create a FeatureAttributionSequenceOutput with garbled byte tokens
    # (simulating what Qwen tokenizer's convert_ids_to_tokens returns)
    garbled_source_tokens = [
        TokenWithId("ä½\xa0å¥½", 100),  # Garbled bytes for "你好"
        TokenWithId("ä¸\x96ç\x95\x8c", 101),  # Garbled bytes for "世界"
    ]
    garbled_target_tokens = [
        TokenWithId("<s>", -1),  # Special token (should be skipped, id < 0)
        TokenWithId("ä½\xa0å¥½", 100),
        TokenWithId("ï¼\x81", 102),  # Garbled bytes for "！"
    ]

    seq_output = FeatureAttributionSequenceOutput(
        source=garbled_source_tokens,
        target=garbled_target_tokens,
        source_attributions=torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]),
        target_attributions=None,
        attr_pos_start=1,
        attr_pos_end=3,
    )

    # Call decode_tokens
    result = seq_output.decode_tokens(mock_tokenizer)

    # Verify method returns self for chaining
    assert result is seq_output

    # Verify source tokens are decoded
    assert seq_output.source[0].token == "你好"
    assert seq_output.source[0].id == 100
    assert seq_output.source[1].token == "世界"
    assert seq_output.source[1].id == 101

    # Verify target tokens are decoded (except special token with id < 0)
    assert seq_output.target[0].token == "<s>"  # Unchanged (id = -1)
    assert seq_output.target[0].id == -1
    assert seq_output.target[1].token == "你好"
    assert seq_output.target[1].id == 100
    assert seq_output.target[2].token == "！"
    assert seq_output.target[2].id == 102


def test_decode_tokens_on_attribution_output(saliency_gpt2_model_tiny):
    """Test decode_tokens method on FeatureAttributionOutput applies to all sequences."""
    out = saliency_gpt2_model_tiny.attribute(
        ["Hello world", "Test input"],
        ["Hello world output", "Test input result"],
        show_progress=False,
        device="cpu",
    )

    # Store original token strings
    original_tokens_0 = [t.token for t in out.sequence_attributions[0].target]
    original_tokens_1 = [t.token for t in out.sequence_attributions[1].target]

    # Call decode_tokens
    result = out.decode_tokens(saliency_gpt2_model_tiny.tokenizer)

    # Verify method returns self for chaining
    assert result is out

    # Verify tokens are still present (decode should work, even if tokens look similar for ASCII)
    assert len(out.sequence_attributions[0].target) == len(original_tokens_0)
    assert len(out.sequence_attributions[1].target) == len(original_tokens_1)

    # For GPT-2 (BPE tokenizer), decoded tokens should be valid strings
    for seq_attr in out.sequence_attributions:
        for token in seq_attr.target:
            assert isinstance(token.token, str)
