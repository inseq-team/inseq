import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def tiny_model(family="llama", kv_heads=2):
    config_cls, model_cls = (
        (LlamaConfig, LlamaForCausalLM) if family == "llama" else (MistralConfig, MistralForCausalLM)
    )
    config = config_cls(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return model_cls(config).eval()


@pytest.fixture
def model():
    return tiny_model()


@pytest.fixture
def tokenizer():
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "the": 4,
        "movie": 5,
        "is": 6,
        "good": 7,
        "bad": 8,
        "very": 9,
        "positive": 10,
        "negative": 11,
    }
    vocab.update({f"token{i}": i for i in range(12, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side="left",
    )
