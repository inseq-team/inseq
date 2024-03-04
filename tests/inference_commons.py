import json
import os

from inseq.data import EncoderDecoderBatch
from inseq.utils import json_advanced_load

from . import TEST_DIR


def get_example_batches():
    dict_batches = json_advanced_load(f"{TEST_DIR}/fixtures/m2m_418M_batches.json")
    dict_batches["batches"] = [batch.torch() for batch in dict_batches["batches"]]
    assert all(isinstance(batch, EncoderDecoderBatch) for batch in dict_batches["batches"])
    return dict_batches


def load_examples() -> dict:
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/huggingface_model.json")
    return json.load(open(file))
