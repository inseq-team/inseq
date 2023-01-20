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

import json
from collections import OrderedDict
from json import JSONEncoder
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

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

EncodableObject = TypeVar("EncodableObject")
DecodableObject = TypeVar("DecodableObject")


def class_instance_encode(obj: EncodableObject, use_primitives: bool = True, **kwargs):
    """
    Encodes a class instance to json. Note that it can only be recovered if the environment allows the class to be
    imported in the same way.
    """
    if isinstance(obj, (list, dict)):
        return obj
    if hasattr(obj, "__class__") and hasattr(obj, "__dict__"):
        if not hasattr(obj, "__new__"):
            raise TypeError(f"class '{obj.__class__}' does not have a __new__ method; ")
        if isinstance(obj, type(lambda: 0)):
            raise TypeError(f"instance '{obj}' of class '{obj.__class__}' cannot be encoded, it is a function.")
        try:
            obj.__new__(obj.__class__)
        except TypeError as err:
            raise TypeError(
                f"instance '{obj}' of class '{obj.__class__}' cannot be encoded, perhaps because its"
                " __new__ method cannot be called because it requires extra parameters"
            ) from err
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


def ndarray_encode(
    obj: EncodableObject,
    use_primitives: bool = True,
    ndarray_compact: Optional[bool] = None,
    compression: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Encodes numpy ``ndarray`` as lists with meta data.
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


ENCODE_HOOKS = [class_instance_encode, ndarray_encode]


class AttributionSerializer(JSONEncoder):
    def __init__(
        self,
        encoders: Optional[List[Callable]] = None,
        use_primitives: bool = True,
        ndarray_compact: Optional[bool] = None,
        compression: bool = False,
        **json_kwargs,
    ):
        self.encoders = []
        if encoders:
            self.encoders = list(encoders)
        self.use_primitives = use_primitives
        self.ndarray_compact = ndarray_compact
        self.compression = compression
        super().__init__(**json_kwargs)

    def default(self, obj: EncodableObject, *args, **kwargs):
        """
        This is the method of JSONEncoders that is called for each object; it calls all the encoders with the previous
        one's output used as input. Works for Encoder instances, but they are expected not to throw ``TypeError`` for
        unrecognized types (the super method does that by default). It never calls the ``super`` method so if there are
        non-primitive types left at the end, you'll get an encoding error.
        """
        prev_id = id(obj)
        for encoder in self.encoders:
            obj = encoder(
                obj,
                use_primitives=self.use_primitives,
                ndarray_compact=self.ndarray_compact,
                compression=self.compression,
            )
        if id(obj) == prev_id:
            raise TypeError(
                f"Object of type {type(obj)} could not be encoded by {self.__class__.__name__} using encoders"
                f" [{', '.join(str(encoder) for encoder in self.encoders)}]."
            )
        return obj


def json_advanced_dumps(
    obj: EncodableObject,
    sort_keys: bool = True,
    encoders: List[Callable] = ENCODE_HOOKS,
    use_primitives: bool = True,
    allow_nan: bool = True,
    ndarray_compact: Optional[bool] = None,
    compression: bool = False,
    **jsonkwargs,
) -> str:
    """Dumps a complex object containing classes and arrays object to a string.

    Args:
        obj (:obj:`Any`):
            Object to be dumped to file.
        sort_keys (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to object fields should be sorted in the serialized output.
        encoders (:obj:`list` of callables, *optional*):
            A list of encoders to run on the object fields for encoding it to JSON-compatible format. By default,
            encoders that serialize classes and numpy arrays are used.
        use_primitives (:obj:`bool`, *optional*, defaults to ``False``):
            If specified, encoders will use primitive types instead of special formats for classes and numpy arrays.
            Note that this will not allow for decoding the object back to its original form.
        allow_nan (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to allow NaN values in the serialized output.
        ndarray_compact (:obj:`bool`, *optional*, defaults to ``None``):
            Whether to use compact storage for numpy arrays. If ``None``, arrays are serialized as lists.
        compression (:obj:`bool`, *optional*, defaults to ``False``):
            Whether to compress the serialized output using GZip.
        **jsonkwargs: Additional keyword arguments passed to :func:`json.dumps`.

    Returns:
        The dumped object in string format.
    """
    combined_encoder = AttributionSerializer(
        encoders=encoders,
        use_primitives=use_primitives,
        sort_keys=sort_keys,
        allow_nan=allow_nan,
        ndarray_compact=ndarray_compact,
        compression=compression,
        **jsonkwargs,
    )
    return combined_encoder.encode(obj)


@save_to_file
def json_advanced_dump(
    obj: EncodableObject,
    sort_keys: bool = True,
    encoders: List[Callable] = ENCODE_HOOKS,
    use_primitives: bool = False,
    allow_nan: bool = True,
    ndarray_compact: Optional[bool] = None,
    compression: bool = False,
    **jsonkwargs,
) -> str:
    """Dumps a complex object containing classes and arrays object to a file.

    Args:
        obj (:obj:`Any`):
            Object to be dumped to file.
        fp (:obj:`str` or :obj:`os.PathLike`):
            File path to which the object should be dumped.
        sort_keys (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to object fields should be sorted in the serialized output.
        encoders (:obj:`list` of callables, *optional*):
            A list of encoders to run on the object fields for encoding it to JSON format. By default, encoders that
            serialize classes and numpy arrays are used.
        use_primitives (:obj:`bool`, *optional*, defaults to ``False``):
            If specified, encoders will use primitive types instead of special formats for classes and numpy arrays.
            Note that this will not allow for decoding the object back to its original form.
        allow_nan (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to allow NaN values in the serialized output.
        ndarray_compact (:obj:`bool`, *optional*, defaults to ``None``):
            Whether to use compact storage for numpy arrays. If ``None``, arrays are serialized as lists.
        compression (:obj:`bool`, *optional*, defaults to ``False``):
            Whether to compress the serialized output using GZip.
        force_flush (:obj:`bool`, *optional*, defaults to ``False``):
            Whether to force flushing the file after writing.
        return_output (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to return the serialized output as a string.
        **jsonkwargs: Additional keyword arguments passed to :func:`json.dumps`.

    Returns:
        The dumped object in string format.
    """
    if isinstance(obj, str) or hasattr(obj, "write"):
        raise ValueError("dump arguments are in the wrong order: provide the data to be serialized before file handle")
    txt = json_advanced_dumps(
        obj,
        sort_keys=sort_keys,
        encoders=encoders,
        use_primitives=use_primitives,
        allow_nan=allow_nan,
        ndarray_compact=ndarray_compact,
        compression=compression,
        **jsonkwargs,
    )
    return txt


def ndarray_hook(dct: Any, **kwargs) -> DecodableObject:
    """
    Replace any numpy arrays previously encoded by NumpyEncoder to their proper shape, data type and data.

    Args:
        dct: The object to be decoded. Will be processed by the hook if it is a dictionary containing the attribute
            ``__ndarray__`` provided by ``ndarray_encode``.
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


def class_instance_hook(dct: Any, cls_lookup_map: Optional[Dict[str, type]] = None, **kwargs) -> DecodableObject:
    """
    This hook tries to convert json encoded by class_instance_encoder back to it's original instance.
    It only works if the environment is the same, e.g. the class is similarly importable and hasn't changed.

    Args:
        dct:
            The object to be decoded. Will be processed by the hook if it is a dictionary containing the attribute
            ``__instance_type__`` provided by ``class_instance_encode``.
        cls_lookup_map (:obj:`dict`, *optional*):
            A dictionary mapping class names to classes. This is used to look up classes when decoding class instances.
            This is useful when the class is not directly importable in the current environment.
    """
    if not isinstance(dct, dict):
        return dct
    if "__instance_type__" not in dct:
        return dct
    mod, name = dct["__instance_type__"]
    curr_class = get_cls_from_instance_type(mod, name, cls_lookup_map=cls_lookup_map)
    try:
        obj = curr_class.__new__(curr_class)
    except TypeError as err:
        raise TypeError(
            f"problem while decoding instance of '{name}'; this instance has a special __new__ method"
        ) from err
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
        """Applies all hooks to the map"""
        map = self.map_type(pairs)
        for hook in self.hooks:
            map = hook(map, cls_lookup_map=self.cls_lookup_map)
        return map


DECODE_HOOKS = [class_instance_hook, ndarray_hook]


def json_advanced_loads(
    string: str,
    ordered: bool = False,
    decompression: bool = False,
    hooks: List[Callable] = DECODE_HOOKS,
    cls_lookup_map: Optional[Dict[str, type]] = None,
    **jsonkwargs,
) -> Any:
    """Load a complex object containing classes and arrays object from a string.

    Args:
        string (:obj:`str`):
            String to be loaded into an object.
        ordered (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to use an :obj:`OrderedDict` to store the loaded data in the original order.
        decompression (:obj:`bool`, *optional*, defaults to ``False``):
            Whether to decompress the file with GZip before loading it.
        hooks (:obj:`list` of callables, *optional*):
            A list of hooks to run on the loaded data for decoding. By default hooks to deserialize classes and numpy
            arrays are used.
        cls_lookup_map (:obj:`dict`, *optional*):
            A dictionary mapping class names to classes. This is used to look up classes when decoding class instances.
            This is useful when the class is not directly importable in the current environment.
        **jsonkwargs: Additional keyword arguments passed to :func:`json.loads`.

    Returns:
        The loaded object.
    """
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


def json_advanced_load(
    fp: Union[str, bytes, PathLike],
    ordered: bool = True,
    decompression: bool = False,
    hooks: List[Callable] = DECODE_HOOKS,
    cls_lookup_map: Optional[Dict[str, type]] = None,
    **jsonkwargs,
) -> Any:
    """Load a complex object containing classes and arrays from a JSON file.

    Args:
        fp (:obj:`str`, :obj:`bytes`, or :obj:`os.PathLike`):
            Path to the file to load.
        ordered (:obj:`bool`, *optional*, defaults to ``True``):
            Whether to use an :obj:`OrderedDict` to store the loaded data in the original order.
        decompression (:obj:`bool`, *optional*, defaults to ``False``):
            Whether to decompress the file with GZip before loading it.
        hooks (:obj:`list` of callables, *optional*):
            A list of hooks to run on the loaded data for decoding. By default hooks to deserialize classes and numpy
            arrays are used.
        cls_lookup_map (:obj:`dict`, *optional*):
            A dictionary mapping class names to classes. This is used to look up classes when decoding class instances.
            This is useful when the class is not directly importable in the current environment.
        **jsonkwargs: Additional keyword arguments passed to :func:`json.loads`.

    Returns:
        The loaded object.
    """
    try:
        if isinstance(fp, (PathLike, bytes, str)):
            with open(fp, "rb" if decompression else "r") as fh:
                string = fh.read()
        else:
            string = fp.read()
    except UnicodeDecodeError as err:
        raise Exception(
            "There was a problem decoding the file content. A possible reason is that the file is not "
            "opened  in binary mode; be sure to set file mode to something like 'rb'."
        ) from err
    return json_advanced_loads(
        string=string,
        ordered=ordered,
        decompression=decompression,
        hooks=hooks,
        cls_lookup_map=cls_lookup_map,
        **jsonkwargs,
    )
