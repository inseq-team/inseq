from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import functools
import logging
import os
import pickle
from contextlib import contextmanager
from functools import wraps
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


def cache_results(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    @wraps(func)
    def cache_results_wrapper(
        cache_dir: str,
        cache_filename: str,
        save_cache: bool = True,
        overwrite_cache: bool = False,
        *args,
        **kwargs,
    ):
        cache_dir = os.path.expanduser(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if os.path.exists(cache_filename) and not overwrite_cache:
            logger.info(f"Loading cached objects from {cache_filename}")
            with open(cache_filename, "rb") as f:
                cached = pickle.load(f)
        else:
            logger.info(f"Cached objects not found in {cache_filename}. Computing...")
            cached = func(*args, **kwargs)
            if save_cache:
                with open(cache_filename, "wb") as f:
                    pickle.dump(cached, f)
        return cached

    return cache_results_wrapper


def ordinal_str(n: int):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(
        4 if 10 <= n % 100 < 20 else n % 10, "th"
    )


def rgetattr(obj, attr, *args):
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
