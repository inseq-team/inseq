import json

import torch
from transformers import AutoTokenizer

from ..base import BaseRationalizer


def serialize_rational(
    filename: str,
    id: int,
    token_inputs: torch.Tensor,
    token_target: torch.Tensor,
    position_rational: torch.Tensor,
    tokenizer: AutoTokenizer,
    important_score: torch.Tensor,
    comments: dict = None,
    compact: bool = False,
    trace_rationalizer: BaseRationalizer = None,
    trace_batch_idx: int = 0,
    schema_file: str = "../docs/rationalization.schema.json"
) -> None:
    """Serialize rationalization result to a json file
    
    Args:
        filename: Filename to store json file
        id: id of the record
        token_inputs: token_inputs [sequence]
        token_target: token_target [1]
        position_rational: position of rational tokens [rational]
        tokenizer: A Huggingface AutoTokenizer
        important_score: final important score of tokens [sequence]
        comments: (Optional) A dictionary of comments
        compact: Whether store json file in a compact style
        trace_rationalizer: (Optional) A Rationalizer with trace started to store trace information
        trace_batch_idx: trace index in the batch, if applicable
        schema_file: location of the json schema file

    """
    data = {
        "$schema": schema_file,
        "id": id,
        "input-text": [tokenizer.decode([i]) for i in token_inputs],
        "input-tokens": [i.item() for i in token_inputs],
        "target-text": tokenizer.decode([token_target]),
        "target-token": token_target.item(),
        "rational-size": position_rational.shape[0],
        "rational-positions": [i.item() for i in position_rational],
        "rational-text": [tokenizer.decode([i]) for i in token_inputs[position_rational]],
        "rational-tokens": [i.item() for i in token_inputs[position_rational]],
    }

    if important_score != None:
        data["importance-scores"] = [i.item() for i in important_score]

    if comments:
        data["comments"] = comments

    if trace_rationalizer:
        trace = {
            "importance-scores": [ [ v.item() for v in i[trace_batch_idx] ] for i in trace_rationalizer.importance_score_evaluator.trace_importance_score ],
            "target-likelihood-original": trace_rationalizer.importance_score_evaluator.trace_target_likelihood_original[trace_batch_idx].item(),
            "target-likelihood": [ i[trace_batch_idx].item() for i in trace_rationalizer.importance_score_evaluator.stopping_condition_evaluator.trace_target_likelihood ]
        }
        data["trace"] = trace

    indent = None if compact else 4
    json_str = json.dumps(data, indent=indent)

    with open(filename, 'w') as f_output:
        f_output.write(json_str)
