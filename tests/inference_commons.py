import json_tricks

from inseq.data import EncoderDecoderBatch

from . import TEST_DIR


def get_example_batches():
    dict_batches = json_tricks.load(f"{TEST_DIR}/fixtures/m2m_418M_batches.json")
    dict_batches["batches"] = [batch.torch() for batch in dict_batches["batches"]]
    assert all(isinstance(batch, EncoderDecoderBatch) for batch in dict_batches["batches"])
    return dict_batches
