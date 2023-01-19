# Code adapted from json-tricks codebase (https://github.com/mverleg/pyjson_tricks)
#
# LICENSE: BSD-3-Clause
#
# Copyright (c) 2022 Mark V. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import json
from collections import OrderedDict
from json import JSONEncoder
from os import PathLike

from numpy import generic, ndarray

from ..utils import (
    bin_str_to_ndarray,
    get_cls_from_instance_type,
    get_module_name_from_object,
    gzip_decompress,
    hashodict,
    lists_of_numbers_to_ndarray,
    ndarray_to_bin_str,
    save_to_file,
    scalar_to_numpy,
)


T = TypeVar("T")


def class_instance_encode(obj: T, use_primitives: bool = True, **kwargs):
    """
    Encodes a class instance to json. Note that it can only be recovered if the environment allows the class to be
    imported in the same way.
    """
    if isinstance(obj, list) or isinstance(obj, dict):
        return obj
    if hasattr(obj, "__class__") and hasattr(obj, "__dict__"):
        if not hasattr(obj, "__new__"):
            raise TypeError(f"class '{obj.__class__}' does not have a __new__ method; ")
        if isinstance(obj, type(lambda: 0)):
            raise TypeError(f"instance '{obj}' of class '{obj.__class__}' cannot be encoded, it is a function.")
        try:
            obj.__new__(obj.__class__)
        except TypeError:
            raise TypeError(
                f"instance '{obj}' of class '{obj.__class__}' cannot be encoded, perhaps because its"
                " __new__ method cannot be called because it requires extra parameters"
            )
        mod = get_module_name_from_object(obj)
        name = obj.__class__.__name__
        if hasattr(obj, "__json_encode__"):
            attrs = obj.__json_encode__()
            if use_primitives or not isinstance(attrs, dict):
                return attrs
            else:
                return hashodict((("__instance_type__", (mod, name)), ("attributes", attrs)))
        dct = hashodict([("__instance_type__", (mod, name))])
        if hasattr(obj, "__dict__"):
            dct["attributes"] = hashodict(obj.__dict__)
        if use_primitives:
            attrs = dct.get("attributes", {})
            return attrs
        else:
            return dct
    return obj


def numpy_encode(
    obj: T, use_primitives: bool = True, ndarray_compact: Optional[bool] = None, compression: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    Encodes numpy `ndarray`s as lists with meta data.
    Encodes numpy scalar types as Python equivalents. Special encoding is not possible,
    because float64 is a subclass of primitives, which never reach the encoder.
    """
    if isinstance(obj, ndarray):
        if use_primitives:
            return obj.tolist()
        else:
            # Property 'ndarray_compact' may also be an integer, in which case it's the number of
            # elements from which compact storage is used.
            if isinstance(ndarray_compact, int) and not isinstance(ndarray_compact, bool):
                ndarray_compact = obj.size >= ndarray_compact
            if ndarray_compact:
                # If the overall json file is compressed, then don't compress the array.
                data_json = ndarray_to_bin_str(obj, do_compress=not compression)
            else:
                data_json = obj.tolist()
            dct = hashodict(
                (
                    ("__ndarray__", data_json),
                    ("dtype", str(obj.dtype)),
                    ("shape", obj.shape),
                )
            )
            if len(obj.shape) > 1:
                dct["Corder"] = obj.flags["C_CONTIGUOUS"]
            return dct
    elif isinstance(obj, generic):
        return obj.item()
    return obj


ENCODE_HOOKS = [class_instance_encode, numpy_encode]


class AttributionSerializer(JSONEncoder):
    def __init__(
        self,
        obj_encoders: Optional[List[Callable]] = None,
        use_primitives: bool = True,
        ndarray_compact: Optional[bool] = None,
        compression: bool = False,
        **json_kwargs,
    ):
        self.obj_encoders = []
        if obj_encoders:
            self.obj_encoders = list(obj_encoders)
        self.use_primitives = use_primitives
        self.ndarray_compact = ndarray_compact
        self.compression = compression
        super().__init__(**json_kwargs)

    def default(self, obj, *args, **kwargs):
        """
        This is the method of JSONEncoders that is called for each object; it calls all the encoders with the previous
        one's output used as input. It works for Encoder instances, but they are expected not to throw `TypeError` for
        unrecognized types (the super method does that by default). It never calls the `super` method so if there are
        non-primitive types left at the end, you'll get an encoding error.
        """
        prev_id = id(obj)
        for encoder in self.obj_encoders:
            obj = encoder(
                obj,
                use_primitives=self.use_primitives,
                ndarray_compact=self.ndarray_compact,
                compression=self.compression,
            )
        if id(obj) == prev_id:
            raise TypeError(
                f"Object of type {type(obj)} could not be encoded by {self.__class__.__name__} using encoders"
                f" [{', '.join(str(encoder) for encoder in self.obj_encoders)}]. You can add an encoders for this type"
                " using `extra_obj_encoders`. If you want to skip this object, consider using `fallback_encoders` like"
                "`str` or `lambda o: None`."
            )
        return obj


def attribution_dumps(
    obj: Any,
    sort_keys: bool = True,
    obj_encoders: List[Callable] = ENCODE_HOOKS,
    use_primitives: bool = True,
    allow_nan: bool = True,
    ndarray_compact: Optional[bool] = None,
    compression: bool = False,
    **jsonkwargs,
) -> str:
    combined_encoder = AttributionSerializer(
        obj_encoders=obj_encoders,
        use_primitives=use_primitives,
        sort_keys=sort_keys,
        allow_nan=allow_nan,
        ndarray_compact=ndarray_compact,
        compression=compression,
        **jsonkwargs,
    )
    return combined_encoder.encode(obj)


@save_to_file
def attribution_dump(
    obj: Any,
    sort_keys: bool = True,
    obj_encoders: List[Callable] = ENCODE_HOOKS,
    use_primitives: bool = True,
    allow_nan: bool = True,
    ndarray_compact: Optional[bool] = None,
    compression: bool = False,
    **jsonkwargs,
) -> str:
    if isinstance(obj, str) or hasattr(obj, "write"):
        raise ValueError("dump arguments are in the wrong order: provide the data to be serialized before file handle")
    txt = attribution_dumps(
        obj,
        sort_keys=sort_keys,
        obj_encoders=obj_encoders,
        use_primitives=use_primitives,
        allow_nan=allow_nan,
        ndarray_compact=ndarray_compact,
        compression=compression,
        **jsonkwargs,
    )
    return txt


def numpy_obj_hook(dct, **kwargs):
    """
    Replace any numpy arrays previously encoded by NumpyEncoder to their proper
    shape, data type and data.
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if not isinstance(dct, dict):
        return dct
    if "__ndarray__" not in dct:
        return dct
    order = None
    if "Corder" in dct:
        order = "C" if dct["Corder"] else "F"
    data_json = dct["__ndarray__"]
    shape = tuple(dct["shape"])
    nptype = dct["dtype"]
    if shape:
        if nptype == "object":
            raise ValueError("Cannot decode object arrays. Object arrays not supported.")
        if isinstance(data_json, str):
            return bin_str_to_ndarray(data_json, order, shape, nptype)
        else:
            return lists_of_numbers_to_ndarray(data_json, order, shape, nptype)
    else:
        return scalar_to_numpy(data_json, nptype)


def class_obj_hook(dct, cls_lookup_map: Optional[Dict[str, type]] = None, **kwargs):
    """
    This hook tries to convert json encoded by class_instance_encoder back to it's original instance.
    It only works if the environment is the same, e.g. the class is similarly importable and hasn't changed.
    """
    if not isinstance(dct, dict):
        return dct
    if "__instance_type__" not in dct:
        return dct
    mod, name = dct["__instance_type__"]
    curr_class = get_cls_from_instance_type(mod, name, cls_lookup_map=cls_lookup_map)
    try:
        obj = curr_class.__new__(curr_class)
    except TypeError:
        raise TypeError(f"problem while decoding instance of '{name}'; this instance has a special __new__ method")
    if hasattr(obj, "__json_decode__"):
        properties = {}
        if "attributes" in dct:
            properties.update(dct["attributes"])
        obj.__json_decode__(**properties)
    else:
        if "attributes" in dct:
            obj.__dict__ = dict(dct["attributes"])
    return obj


class AttributionDeserializer:
    """
    Hook that converts json maps to the appropriate python type (dict or OrderedDict)
    and then runs any number of hooks on the individual maps.
    """

    def __init__(
        self,
        ordered: bool = False,
        hooks: Optional[List[Callable]] = None,
        cls_lookup_map: Optional[Dict[str, type]] = None,
    ):
        self.map_type = OrderedDict if ordered else dict
        self.hooks = hooks if hooks else []
        self.cls_lookup_map = cls_lookup_map

    def __call__(self, pairs):
        map = self.map_type(pairs)
        for hook in self.hooks:
            map = hook(map, cls_lookup_map=self.cls_lookup_map)
        return map


DECODE_HOOKS = [class_obj_hook, numpy_obj_hook]


def attribution_loads(
    string: str,
    ordered: bool = False,
    decompression: bool = False,
    hooks: List[Callable] = DECODE_HOOKS,
    cls_lookup_map: Optional[Dict[str, type]] = None,
    **jsonkwargs,
) -> Any:
    if decompression:
        string = gzip_decompress(string).decode("UTF-8")
    if not isinstance(string, str):
        raise TypeError(f"The input was of non-string type '{type(string)}' in `load(s)`. ")
    hook = AttributionDeserializer(
        ordered=ordered,
        cls_lookup_map=cls_lookup_map,
        hooks=hooks,
    )
    return json.loads(string, object_pairs_hook=hook, **jsonkwargs)


def attribution_load(
    fp: Union[str, bytes, PathLike],
    ordered: bool = True,
    decompression: bool = False,
    hooks: List[Callable] = DECODE_HOOKS,
    cls_lookup_map: Optional[Dict[str, type]] = None,
    **jsonkwargs,
) -> Any:
    try:
        if isinstance(fp, str) or isinstance(fp, bytes) or isinstance(fp, PathLike):
            with open(fp, "rb" if decompression else "r") as fh:
                string = fh.read()
        else:
            string = fp.read()
    except UnicodeDecodeError:
        raise Exception(
            "There was a problem decoding the file content. A possible reason is that the file is not "
            "opened  in binary mode; be sure to set file mode to something like 'rb'."
        )
    return attribution_loads(
        string=string,
        ordered=ordered,
        decompression=decompression,
        hooks=hooks,
        cls_lookup_map=cls_lookup_map,
        **jsonkwargs,
    )
