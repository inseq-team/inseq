from typing import Any, Callable


def set_hook(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def set_hook_wrapper(self, **kwargs):
        f(self, **kwargs)
        self.attribution_model.is_hooked = True

    return set_hook_wrapper


def unset_hook(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def unset_hook_wrapper(self, **kwargs):
        f(self, **kwargs)
        self.attribution_model.is_hooked = False

    return unset_hook_wrapper
