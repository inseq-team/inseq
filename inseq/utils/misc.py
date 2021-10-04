from typing import Any, Optional, Sequence

from contextlib import contextmanager


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


def ordinal_str(n: int):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(
        4 if 10 <= n % 100 < 20 else n % 10, "th"
    )
