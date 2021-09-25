from typing import Callable


def set_hook(f) -> Callable:
    def set_hook_wrapper(self):
        f(self)
        self.attribution_model.is_hooked = True

    return set_hook_wrapper


def unset_hook(f) -> Callable:
    def unset_hook_wrapper(self):
        f(self)
        self.attribution_model.is_hooked = False

    return unset_hook_wrapper
