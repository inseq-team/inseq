from typing import Any, Callable, Dict, Optional, Sequence

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
        and (len(l) > 3 or any([len(sl) > 20 for sl in l]))
    ) or len(l) > 20:
        return f"list with {len(l)} elements"
    return f"list with {len(l)} elements: {l}"


def extract_signature_args(
    full_args: Dict[str, Any],
    func: Callable[[Any], Any],
    exclude_args: Optional[Sequence[str]] = None,
):
    return {
        k: v
        for k, v in full_args.items()
        if k in signature(func).parameters
        and (exclude_args is None or k not in exclude_args)
    }


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
