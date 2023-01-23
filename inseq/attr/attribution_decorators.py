# Copyright 2021 The Inseq Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Decorators for attribution methods. """

import logging
from functools import wraps
from typing import Any, Callable, List, Optional, Sequence

from ..data.data_utils import TensorWrapper

logger = logging.getLogger(__name__)


def set_hook(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Sets the status of the attribution model associated to the function
    to is_hooked = True.
    Required to decorate the hook functions in subclasses.
    """

    @wraps(f)
    def set_hook_wrapper(self, **kwargs):
        f(self, **kwargs)
        self.attribution_model.is_hooked = True

    return set_hook_wrapper


def unset_hook(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Sets the status of the attribution model associated to the function
    to is_hooked = False.
    Required to decorate the unhook functions in subclasses.
    """

    @wraps(f)
    def unset_hook_wrapper(self, **kwargs):
        f(self, **kwargs)
        self.attribution_model.is_hooked = False

    return unset_hook_wrapper


def batched(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that enables batching of the args
    """

    @wraps(f)
    def batched_wrapper(self, *args, batch_size: Optional[int] = None, **kwargs):
        def get_batched(bs: Optional[int], seq: Sequence[Any]) -> List[List[Any]]:
            if isinstance(seq, str):
                seq = [seq]
            if isinstance(seq, list):
                return [seq[i : i + bs] for i in range(0, len(seq), bs)]  # noqa
            if isinstance(seq, tuple):
                return list(zip(*[get_batched(bs, s) for s in seq]))
            elif isinstance(seq, TensorWrapper):
                return [seq.slice_batch(slice(i, i + bs)) for i in range(0, len(seq), bs)]  # noqa
            else:
                raise TypeError(f"Unsupported type {type(seq)} for batched attribution computation.")

        if batch_size is None:
            return [f(self, *args, **kwargs)]
        batched_args = [get_batched(batch_size, arg) for arg in args]
        len_batches = len(batched_args[0])
        assert all(len(batch) == len_batches for batch in batched_args)
        output = []
        zipped_batched_args = zip(*batched_args) if len(batched_args) > 1 else [(x,) for x in batched_args[0]]
        for i, batch in enumerate(zipped_batched_args):
            logger.debug(f"Batching enabled: processing batch {i + 1} of {len_batches}...")
            output.append(f(self, *batch, **kwargs))
        return output

    return batched_wrapper
