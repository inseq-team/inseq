from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import functools
import logging
import numbers
from contextlib import contextmanager
from inspect import signature
from itertools import dropwhile

from torch import Tensor

from .errors import LengthMismatchError
from .typing import TextInput, TokenWithId


logger = logging.getLogger(__name__)


def identity_fn(x, **kwargs):
    return x


@contextmanager
def optional(condition, context_manager, alternative_fn=None, **alternative_fn_kwargs):
    if condition:
        with context_manager:
            yield
    else:
        if alternative_fn is not None:
            alternative_fn(**alternative_fn_kwargs)
        yield


def _pretty_list_contents(l: Sequence[Any]) -> str:
    quote = f"""{"'" if l and type(l[0]) in [str, TokenWithId] else ""}"""
    return (
        quote
        + f"{quote}, {quote}".join(
            [
                f"{' ' if isinstance(v, numbers.Number) and v >= 0 else ''}"
                + (f"{v:.2f}" if isinstance(v, float) else f"{v}")
                for v in l
            ]
        )
        + quote
    )


def _pretty_list(l: Optional[Sequence[Any]], lpad: int = 8) -> str:
    if all([isinstance(x, list) for x in l]):
        line_sep = f" ],\n{' ' * lpad}[ "
        contents = " " * lpad + "[ " + line_sep.join([_pretty_list_contents(subl) for subl in l]) + " ]"
    else:
        if all([hasattr(x, "to_dict") for x in l]):
            contents = ",\n".join([f"{' ' * lpad + x.__class__.__name__}({pretty_dict(x.to_dict(), lpad)}" for x in l])
        else:
            contents = " " * lpad + _pretty_list_contents(l)
    return "[\n" + contents + f"\n{' ' * (lpad - 4)}]"


def pretty_list(l: Optional[Sequence[Any]], lpad: int = 8) -> str:
    if l is None:
        return "None"
    if len(l) == 0:
        return "list with 0 elements"
    out_txt = f"list with {len(l)} elements of type {l[0].__class__.__name__}"
    if all([isinstance(x, list) for x in l]):
        out_txt = f"list with {len(l)} sub-lists"
        if any([len(sl) > 20 for sl in l]) or len(l) > 15:
            return out_txt
    if len(l) > 20:
        return out_txt
    return f"{out_txt}:{_pretty_list(l, lpad)}"


def pretty_tensor(t: Optional[Tensor] = None, lpad: int = 8) -> str:
    if t is None:
        return "None"
    if len(t.shape) > 3 or any([x > 20 for x in t.shape]):
        return f"{t.dtype} tensor of shape {list(t.shape)} on {t.device}"
    else:
        out_list = t.tolist()
        out_list = _pretty_list(out_list, lpad) if isinstance(out_list, list) else out_list
        return f"{t.dtype} tensor of shape {list(t.shape)} on {t.device}: {out_list}"


def pretty_dict(d: Dict[str, Any], lpad: int = 4) -> str:
    out_txt = "{\n"
    for k, v in d.items():
        out_txt += f"{' ' * lpad}{k}: "
        if isinstance(v, list) or isinstance(v, tuple):
            out_txt += pretty_list(v, lpad + 4)
        elif isinstance(v, Tensor):
            out_txt += pretty_tensor(v, lpad + 4)
        elif isinstance(v, dict):
            out_txt += pretty_dict(v, lpad + 4)
        elif hasattr(v, "to_dict"):
            out_txt += pretty_dict(v.to_dict(), lpad + 4)
        else:
            out_txt += "None" if v is None else str(v)
        out_txt += ",\n"
    return out_txt + f"{' ' * (lpad - 4)}}}"


def extract_signature_args(
    full_args: Dict[str, Any],
    func: Callable[[Any], Any],
    exclude_args: Optional[Sequence[str]] = None,
    return_remaining: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
    extracted_args = {
        k: v
        for k, v in full_args.items()
        if k in signature(func).parameters and (exclude_args is None or k not in exclude_args)
    }
    if return_remaining:
        return extracted_args, {k: v for k, v in full_args.items() if k not in extracted_args}
    return extracted_args


def ordinal_str(n: int):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")


def rgetattr(obj, attr, *args):
    """Recursively access attributes from nested classes

    E.g. rgetattr(attr_model, 'model.model.decoder.layers[4].self_attn')
    >> MarianAttention(
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
    )
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    attr = attr.replace("[", ".").replace("]", "")
    return functools.reduce(_getattr, [obj] + attr.split("."))


def find_char_indexes(strings: Sequence[str], char: str = " "):
    """Finds the indexes of a character in a list of strings."""
    whitespace_indexes = []
    for sent in strings:
        idx = 0
        curr_idxs = []
        for token in sent.split(char):
            idx += len(token)
            curr_idxs.append(idx)
            idx += 1
        whitespace_indexes.append(curr_idxs)
    return whitespace_indexes


def pad(seq: Sequence[Sequence[Any]], pad_id: Any):
    """Pads a list of sequences to the same length."""
    max_len = max(len(x) for x in seq)
    seq = [x + [pad_id] * (max_len - len(x)) for x in seq]
    return seq


def drop_padding(seq: Sequence[Any], pad_id: Any):
    if pad_id is None:
        return seq
    return list(reversed(list(dropwhile(lambda x: x == pad_id, reversed(seq)))))


def isnotebook():
    """Returns true if code is being executed in a notebook, false otherwise

    Currently supported: Jupyter Notebooks, Google Colab
    To validate: Kaggle Notebooks, JupyterLab
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        module = get_ipython().__class__.__module__
        if shell == "ZMQInteractiveShell" or module == "google.colab._shell":
            return True  # Jupyter notebook, Google Colab or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
    except ModuleNotFoundError:
        return False  # IPython not installed


def format_input_texts(
    texts: TextInput,
    ref_texts: Optional[TextInput] = None,
) -> Tuple[List[str], List[str]]:
    texts = [texts] if isinstance(texts, str) else texts
    reference_texts = [ref_texts] if isinstance(ref_texts, str) else ref_texts
    if reference_texts and len(texts) != len(reference_texts):
        raise LengthMismatchError(
            "Length mismatch for texts and reference_texts."
            "Input length: {}, reference length: {} ".format(len(texts), len(reference_texts))
        )
    return texts, reference_texts


def aggregate_token_sequence(token_sequence, spans):
    if not spans:
        return token_sequence
    out_sequence = []
    span_start_idxs = [span[0] for span in spans]
    curr_idx = 0
    for tok_idx, token in enumerate(token_sequence):
        if tok_idx < curr_idx:
            continue
        if curr_idx in span_start_idxs:
            end_idx = spans[span_start_idxs.index(curr_idx)][1]
            # We use -1 as token index to indicate the token is product of an aggregation
            # (i.e. not contained in the original vocabulary)
            out_sequence.append(TokenWithId("".join([t.token for t in token_sequence[curr_idx:end_idx]]), -1))
            curr_idx = end_idx
        else:
            out_sequence.append(token)
            curr_idx += 1
    return out_sequence


def aggregate_token_pair(tokens: List[TokenWithId], other_tokens: List[TokenWithId]):
    if not other_tokens:
        return tokens
    out_tokens = []
    for tok, other in zip(tokens, other_tokens):
        if tok.token == other.token:
            out_tokens.append(TokenWithId(tok.token, tok.id))
        else:
            out_tokens.append(TokenWithId(tok.token + " → " + other.token, -1))
    return out_tokens
