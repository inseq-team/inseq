import functools
import gzip
import io
import logging
import math
from base64 import standard_b64decode, standard_b64encode
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from inspect import signature
from numbers import Number
from os import PathLike, fsync
from typing import Any

from numpy import asarray, frombuffer
from torch import Tensor

from .errors import LengthMismatchError
from .typing import TextInput, TokenWithId

logger = logging.getLogger(__name__)


@contextmanager
def optional(condition, context_manager, alternative_fn=None, **alternative_fn_kwargs):
    if condition:
        with context_manager:
            yield
    else:
        if alternative_fn is not None:
            alternative_fn(**alternative_fn_kwargs)
        yield


def _pretty_list_contents(l: Sequence[Any]) -> str:
    quote = "'" if l and type(l[0]) in [str, TokenWithId] else ""
    get_space = lambda s: "" if not isinstance(s, Number) or s < 0 else "  " if math.isnan(s) else " "
    return (
        quote
        + f"{quote}, {quote}".join([get_space(v) + (f"{v:.2f}" if isinstance(v, float) else f"{v}") for v in l])
        + quote
    )


def _pretty_list(l: Sequence[Any] | None, lpad: int = 8) -> str:
    if all(isinstance(x, list) for x in l):
        line_sep = f" ],\n{' ' * lpad}[ "
        contents = " " * lpad + "[ " + line_sep.join([_pretty_list_contents(subl) for subl in l]) + " ]"
    elif all(hasattr(x, "to_dict") for x in l):
        contents = ",\n".join(
            [f"{' ' * lpad + x.__class__.__name__}({pretty_dict(x.to_dict(), lpad + 4)})" for x in l]
        )
    else:
        contents = " " * lpad + _pretty_list_contents(l)
    return "[\n" + contents + f"\n{' ' * (lpad - 4)}]"


def pretty_list(l: Sequence[Any] | None, lpad: int = 8) -> str:
    if l is None:
        return "None"
    if len(l) == 0:
        return "list with 0 elements"
    out_txt = f"list with {len(l)} elements of type {l[0].__class__.__name__}"
    if all(isinstance(x, list) for x in l):
        out_txt = f"list with {len(l)} sub-lists"
        if any(len(sl) > 20 for sl in l) or len(l) > 15:
            return out_txt
        if all(isinstance(ssl, list) for sl in l for ssl in sl):
            return out_txt
    if len(l) > 20:
        return out_txt
    return f"{out_txt}: {_pretty_list(l, lpad)}"


def pretty_tensor(t: Tensor | None = None, lpad: int = 8) -> str:
    if t is None:
        return "None"
    if t.ndim > 2 or any(x > 20 for x in t.shape):
        return f"{t.dtype} tensor of shape {list(t.shape)} on {t.device}"
    else:
        out_list = t.tolist()
        out_list = _pretty_list(out_list, lpad) if isinstance(out_list, list) else out_list
        return f"{t.dtype} tensor of shape {list(t.shape)} on {t.device}: {out_list}"


def pretty_dict(d: dict[str, Any], lpad: int = 4) -> str:
    if not d:
        return "{}"
    out_txt = "{\n"
    for k, v in d.items():
        out_txt += f"{' ' * lpad}{k}: "
        if isinstance(v, list | tuple):
            out_txt += pretty_list(v, lpad + 4)
        elif isinstance(v, Tensor):
            out_txt += pretty_tensor(v, lpad + 4)
        elif isinstance(v, dict):
            out_txt += pretty_dict(v, lpad + 4)
        elif hasattr(v, "to_dict") and not isinstance(v, type):
            out_txt += f"{v.__class__.__name__}({pretty_dict(v.to_dict(), lpad + 4)})"
        elif v is None:
            out_txt += "None"
        elif isinstance(v, str):
            out_txt += f'"{v}"'
        else:
            out_txt += str(v)
        out_txt += ",\n"
    return out_txt + f"{' ' * (lpad - 4)}}}"


def extract_signature_args(
    full_args: dict[str, Any],
    func: Callable[[Any], Any],
    exclude_args: Sequence[str] | None = None,
    return_remaining: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    extracted_args = {
        k: v
        for k, v in full_args.items()
        if k in signature(func).parameters and (exclude_args is None or k not in exclude_args)
    }
    if return_remaining:
        return extracted_args, {k: v for k, v in full_args.items() if k not in extracted_args}
    return extracted_args


def ordinal_str(n: int):
    """Converts a number to an ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")


def rgetattr(obj, attr, *args):
    """Recursively access attributes from nested classes.

    E.g. rgetattr(attr_model, 'model.model.decoder.layers[4].self_attn')
    >> MarianAttention(
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
    )
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    attr = attr.replace("[", ".").replace("]", "")
    return functools.reduce(_getattr, [obj] + attr.split("."))


def find_char_indexes(strings: Sequence[str], char: str = " "):
    """Finds the indexes of a character in a list of strings."""
    whitespace_indexes = []
    for sent in strings:
        idx = 0
        curr_idxs = []
        for token in sent.split(char):
            idx += len(token)
            curr_idxs.append(idx)
            idx += 1
        whitespace_indexes.append(curr_idxs)
    return whitespace_indexes


def pad(seq: Sequence[Sequence[Any]], pad_id: Any):
    """Pads a list of sequences to the same length."""
    max_len = max(len(x) for x in seq)
    seq = [x + [pad_id] * (max_len - len(x)) for x in seq]
    return seq


def drop_padding(seq: Sequence[TokenWithId], pad_id: str):
    if pad_id is None:
        return seq
    return [x for x in seq if x.token != pad_id]


def isnotebook():
    """Returns true if code is being executed in a notebook, false otherwise.

    Currently supported: Jupyter Notebooks, Google Colab
    To validate: Kaggle Notebooks, JupyterLab
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        module = get_ipython().__class__.__module__
        if shell == "ZMQInteractiveShell" or module == "google.colab._shell":
            return True  # Jupyter notebook, Google Colab or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
    except ModuleNotFoundError:
        return False  # IPython not installed


def format_input_texts(
    texts: TextInput,
    ref_texts: TextInput | None = None,
    skip_special_tokens: bool = False,
    special_tokens: list[str] = [],
) -> tuple[list[str], list[str]]:
    texts = [texts] if isinstance(texts, str) else texts
    reference_texts = [ref_texts] if isinstance(ref_texts, str) else ref_texts
    if reference_texts and texts and len(texts) != len(reference_texts):
        raise LengthMismatchError(
            "Length mismatch for texts and reference_texts.Input length: {}, reference length: {} ".format(
                len(texts), len(reference_texts)
            )
        )
    if skip_special_tokens:
        for special_token in special_tokens:
            texts = [text.replace(special_token, "") for text in texts]
            if reference_texts is not None:
                reference_texts = [text.replace(special_token, "") for text in reference_texts]
    return texts, reference_texts


def aggregate_token_sequence(token_sequence, spans):
    if not spans:
        return token_sequence
    out_sequence = []
    span_start_idxs = [span[0] for span in spans]
    curr_idx = 0
    for tok_idx, token in enumerate(token_sequence):
        if tok_idx < curr_idx:
            continue
        if curr_idx in span_start_idxs:
            end_idx = spans[span_start_idxs.index(curr_idx)][1]
            # We use -1 as token index to indicate the token is product of an aggregation
            # (i.e. not contained in the original vocabulary)
            out_sequence.append(TokenWithId("".join([t.token for t in token_sequence[curr_idx:end_idx]]), -1))
            curr_idx = end_idx
        else:
            out_sequence.append(token)
            curr_idx += 1
    return out_sequence


def aggregate_token_pair(tokens: list[TokenWithId], other_tokens: list[TokenWithId]):
    if not other_tokens:
        return tokens
    out_tokens = []
    for tok, other in zip(tokens, other_tokens, strict=False):
        if tok.token == other.token:
            out_tokens.append(TokenWithId(tok.token, tok.id))
        else:
            out_tokens.append(TokenWithId(tok.token + " → " + other.token, -1))
    return out_tokens


def gzip_compress(data, compresslevel):
    """Do gzip compression, without the timestamp."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=compresslevel, mtime=0) as fh:
        fh.write(data)
    return buf.getvalue()


def gzip_decompress(data):
    """Do gzip decompression, without the timestamp."""
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        return f.read()


def ndarray_to_bin_str(array, do_compress):
    """From ndarray to base64 encoded, gzipped binary data."""
    assert array.flags["C_CONTIGUOUS"], "only C memory order is (currently) supported for compact ndarray format"

    original_size = array.size * array.itemsize
    header = "b64:"
    data = array.data
    if do_compress:
        small = gzip_compress(data, compresslevel=9)
        if len(small) < 0.9 * original_size and len(small) < original_size - 8:
            header = "b64.gz:"
            data = small
    data = standard_b64encode(data)
    return header + data.decode("ascii")


class hashodict(OrderedDict):
    """This dictionary is hashable. It should NOT be mutated, or all kinds of weird
    bugs may appear. This is not enforced though, it's only used for encoding.
    """

    def __hash__(self):
        return hash(frozenset(self.items()))


def get_module_name_from_object(obj):
    mod = obj.__class__.__module__
    if mod == "__main__":
        mod = None
        logger.warning(
            f"class {obj.__class__} seems to have been defined in the main file; unfortunately this means"
            " that it's module/import path is unknown, so you might have to provide cls_lookup_map when decoding"
        )
    return mod


def save_to_file(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Serializes the function output to a file, performing the required checks."""

    @wraps(f)
    def save_to_file_wrapper(
        obj: Any,
        fp: str | bytes | PathLike = None,
        *args,
        compression: int | bool = None,
        force_flush: bool = False,
        return_output: bool = True,
        **kwargs,
    ) -> Any | None:
        if "compression" in signature(f).parameters:
            kwargs["compression"] = compression
        txt = f(obj, *args, **kwargs)
        if compression:
            if compression is True:
                compression = 5
            txt = txt.encode("UTF-8")
            txt = gzip_compress(txt, compresslevel=compression)
        if isinstance(fp, str):
            if compression:
                fh = open(fp, "wb+")
            else:
                fh = open(fp, "w+")
        else:
            fh = fp
        try:
            if compression and "b" not in getattr(fh, "mode", "b?") and not isinstance(txt, str):
                raise OSError("If compression is enabled, the file must be opened in binary mode.")
            try:
                fh.write(txt)
            except TypeError as err:
                err.args = (
                    err.args[0] + ". A possible reason is that the file is not opened in binary mode; "
                    "be sure to set file mode to something like 'wb'.",
                )
                raise
        finally:
            if force_flush:
                fh.flush()
                try:
                    if fh.fileno() is not None:
                        fsync(fh.fileno())
                except ValueError:
                    pass
            if isinstance(fp, str):
                fh.close()
        if return_output:
            return txt

    return save_to_file_wrapper


def bin_str_to_ndarray(data, order, shape, dtype):
    """From base64 encoded, gzipped binary data to ndarray."""
    assert order in [
        None,
        "C",
    ], f"specifying different memory order is not (yet) supported for binary numpy format (got order = {order})"
    if data.startswith("b64.gz:"):
        data = standard_b64decode(data[7:])
        data = gzip_decompress(data)
    elif data.startswith("b64:"):
        data = standard_b64decode(data[4:])
    else:
        raise ValueError("found numpy array buffer, but did not understand header; supported: b64 or b64.gz")
    data = frombuffer(data, dtype=dtype)
    return data.reshape(shape)


def lists_of_numbers_to_ndarray(data, order, shape, dtype):
    """From nested list of numbers to ndarray."""
    arr = asarray(data, dtype=dtype, order=order)
    if 0 in shape:
        return arr.reshape(shape)
    if shape != arr.shape:
        logger.warning(f"size mismatch decoding numpy array: expected {shape}, got {arr.shape}")
    return arr


def scalar_to_numpy(data, dtype):
    """From scalar value to numpy type."""
    import numpy as nptypes

    dtype = getattr(nptypes, dtype)
    return dtype(data)


def get_cls_from_instance_type(mod, name, cls_lookup_map):
    curr_class = ValueError()
    if mod is None:
        try:
            curr_class = getattr((__import__("__main__")), name)
        except (ImportError, AttributeError) as err:
            if name not in cls_lookup_map:
                raise ImportError(
                    f"class {name} seems to have been exported from the main file, which means "
                    "it has no module/import path set; you need to provide loads argument"
                    f"`cls_lookup_map={{'{name}': Class}}` to locate the class"
                ) from err
            curr_class = cls_lookup_map[name]
    else:
        imp_err = None
        try:
            module = import_module(f"{mod}")
        except ImportError as err:
            imp_err = (
                f"encountered import error '{err}' while importing '{mod}' to decode a json file; perhaps "
                f"it was encoded in a different environment where {mod}.{name} was available"
            )
        else:
            if hasattr(module, name):
                curr_class = getattr(module, name)
            else:
                imp_err = (
                    f"imported '{module}' but could find '{name}' inside while decoding a json file "
                    f"(found {', '.join(attr for attr in dir(module) if not attr.startswith('_'))})"
                )
        if imp_err:
            curr_class = cls_lookup_map.get(name, None)
            if curr_class is None:
                raise ImportError(f"{imp_err}; add the class to `cls_lookup_map={{'{name}': Class}}` argument")
    return curr_class


def clean_tokens(
    tokens: list[str],
    remove_tokens: list[str] = [],
    return_removed_idxs: bool = False,
    replace_chars: dict[str, str] | None = None,
) -> list[str] | tuple[list[str], list[int]]:
    """Removes tokens from a list of tokens and returns the cleaned list and the removed token indexes."""
    clean_tokens = []
    removed_token_idxs = []
    for idx, tok in enumerate(tokens):
        new_tok = tok
        if new_tok in remove_tokens:
            removed_token_idxs += [idx]
        else:
            if replace_chars is not None:
                for k, v in replace_chars.items():
                    new_tok = new_tok.replace(k, v)
            clean_tokens += [new_tok.strip()]
    if return_removed_idxs:
        return clean_tokens, removed_token_idxs
    return clean_tokens


def get_left_padding(text: str):
    """Returns the number of spaces at the beginning of a string."""
    return len(text) - len(text.lstrip())
