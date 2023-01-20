from functools import wraps
from typing import Any, Callable


def unhooked(f: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(f)
    def attribution_free_wrapper(self, *args, **kwargs):
        was_hooked = False
        if self.is_hooked:
            was_hooked = True
            self.attribution_method.unhook()
        out = f(self, *args, **kwargs)
        if was_hooked:
            self.attribution_method.hook()
        return out

    return attribution_free_wrapper
