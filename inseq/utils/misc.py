from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import functools
import logging
from contextlib import contextmanager
from inspect import signature

logger = logging.getLogger(__name__)


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield


def pretty_list(l: Optional[Sequence[Any]]) -> str:
    if l is None:
        return "None"
    elif (
        all([isinstance(x, list) for x in l])
        and (len(l) > 4 or any([len(sl) > 20 for sl in l]))
    ) or len(l) > 20:
        return f"list with {len(l)} elements"
    return f"list with {len(l)} elements: {l}"


def extract_signature_args(
    full_args: Dict[str, Any],
    func: Callable[[Any], Any],
    exclude_args: Optional[Sequence[str]] = None,
    return_remaining: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
    extracted_args = {
        k: v
        for k, v in full_args.items()
        if k in signature(func).parameters
        and (exclude_args is None or k not in exclude_args)
    }
    if return_remaining:
        return extracted_args, {
            k: v for k, v in full_args.items() if k not in extracted_args
        }
    return extracted_args


def ordinal_str(n: int):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(
        4 if 10 <= n % 100 < 20 else n % 10, "th"
    )


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
