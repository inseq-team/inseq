import re
from inspect import getsourcelines
from sys import gettrace, settrace
from typing import Callable, Optional, TypeVar

from torch import nn

StackFrame = TypeVar("StackFrame")


def get_last_variable_assignment_position(
    module: nn.Module,
    varname: str,
    fname: str = "forward",
) -> Optional[int]:
    """Extract the code line number of the last variable assignment for a variable of interest in the specified method
    of a `nn.Module` object.

    Args:
        module (`nn.Module`):
            A PyTorch module containing a method with a variable assignment after which the hook should be executed.
        varname (`str`):
            The name of the variable to use as anchor for the hook.
        fname (`str`, *optional*, defaults to "forward"):
            The name of the method in which the variable assignment should be searched.

    Returns:
        `Optional[int]`: Returns the line number in the file (not relative to the method) of the last variable
        assignment. Returns None if no assignment to the variable was found.
    """
    # Matches any assignment of variable varname
    pattern = rf"^\s*(?:\w+\s*,\s*)*\b{varname}\b\s*(?:,.+\s*)*=\s*[^\W=]+.*$"
    code, startline = getsourcelines(getattr(module, fname))
    line_numbers = [i for i, line in enumerate(code) if re.match(pattern, line)]
    if len(line_numbers) == 0:
        return None
    return line_numbers[-1] + startline + 1


def get_post_variable_assignment_hook(
    module: nn.Module,
    varname: str,
    fname: str = "forward",
    hook_fn: Callable[[StackFrame], None] = lambda **kwargs: None,
    **kwargs,
) -> Callable[[], None]:
    """Creates a hook that is called after the last variable assignment in the specified method of a `nn.Module`.

    This is a hacky method using the ``sys.settrace()`` function to circumvent the limited hook points of Pytorch hooks
    and set a custom hook point dynamically. This approach is preferred to ensure a broader compatibility with Hugging
    Face transformers models that do not provide hook points in their architectures for the moment.

    Args:
        module (`nn.Module`):
            A PyTorch module containing a method with a variable assignment after which the hook should be executed.
        varname (`str`):
            The name of the variable to use as anchor for the hook.
        fname (`str`, *optional*, defaults to "forward"):
            The name of the method in which the variable assignment should be searched.
        hook_fn (`Callable[[FrameType], None]`, *optional*, defaults to lambdaframe):
            A custom hook function that is called after the last variable assignment in the specified method. The first
            parameter is the current frame in the execution at the hook point, and any additional arguments can be
            passed when creating the hook. ``frame.f_locals`` is a dictionary containing all local variables.

    Returns:
        The hook function that can be registered with the module. If hooking the module's ``forward()`` method, the
        hook can be registered with Pytorch native hook methods.
    """
    hook_line_num = get_last_variable_assignment_position(module, varname, fname)
    curr_trace_fn = gettrace()
    if hook_line_num is None:
        raise ValueError(f"Could not find assignment to {varname} in {module}'s {fname}() method")

    def var_tracer(frame, event, arg=None):
        curr_line_num = frame.f_lineno
        curr_func_name = frame.f_code.co_name

        # Matches the first executable line after hook_line_num in the same function of the same module
        if (
            event == "line"
            and curr_line_num >= hook_line_num
            and curr_func_name == fname
            and isinstance(frame.f_locals.get("self"), nn.Module)
            and frame.f_locals.get("self")._get_name() == module._get_name()
        ):
            # Call the custom hook providing the current frame and any additional arguments as context
            hook_fn(frame, **kwargs)
            settrace(curr_trace_fn)
        return var_tracer

    def hook(*args, **kwargs):
        settrace(var_tracer)

    return hook
