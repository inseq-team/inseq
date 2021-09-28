from typing import Any, Callable, Dict, Optional, Sequence

from contextlib import contextmanager
from inspect import signature


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
