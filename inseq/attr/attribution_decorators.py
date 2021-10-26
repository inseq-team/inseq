# Copyright 2021 Gabriele Sarti. All rights reserved.
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

from typing import Any, Callable


def set_hook(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Sets the status of the attribution model associated to the function
    to is_hooked = True.
    Required to decorate the hook functions in subclasses.
    """

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

    def unset_hook_wrapper(self, **kwargs):
        f(self, **kwargs)
        self.attribution_model.is_hooked = False

    return unset_hook_wrapper
