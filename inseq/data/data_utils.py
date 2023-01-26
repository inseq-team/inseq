from copy import deepcopy
from dataclasses import dataclass, fields

import numpy as np
import torch
from torchtyping import TensorType

from ..utils import pretty_dict


@dataclass
class TensorWrapper:
    @staticmethod
    def _getitem(attr, subscript):
        if isinstance(attr, torch.Tensor):
            if len(attr.shape) == 1:
                return attr[subscript]
            if len(attr.shape) >= 2:
                return attr[:, subscript, ...]
        elif isinstance(attr, TensorWrapper):
            return attr[subscript]
        elif isinstance(attr, list) and isinstance(attr[0], list):
            return [seq[subscript] for seq in attr]
        elif isinstance(attr, dict):
            return {key: TensorWrapper._getitem(val, subscript) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _slice_batch(attr, subscript):
        if isinstance(attr, torch.Tensor):
            if len(attr.shape) == 1:
                return attr[subscript]
            if len(attr.shape) >= 2:
                return attr[subscript, ...]
        elif isinstance(attr, (TensorWrapper, list)):
            return attr[subscript]
        elif isinstance(attr, dict):
            return {key: TensorWrapper._slice_batch(val, subscript) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _select_active(attr, mask):
        if isinstance(attr, torch.Tensor):
            if len(attr.shape) <= 1:
                return attr
            else:
                curr_mask = mask.clone()
                if curr_mask.dtype != torch.bool:
                    curr_mask = curr_mask.bool()
                while len(curr_mask.shape) < len(attr.shape):
                    curr_mask = curr_mask.unsqueeze(-1)
                orig_shape = attr.shape[1:]
                return attr.masked_select(curr_mask).reshape(-1, *orig_shape)
        elif isinstance(attr, TensorWrapper):
            return attr.select_active(mask)
        elif isinstance(attr, list):
            return [val for i, val in enumerate(attr) if mask.tolist()[i]]
        elif isinstance(attr, dict):
            return {key: TensorWrapper._select_active(val, mask) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _to(attr, device: str):
        if isinstance(attr, (torch.Tensor, TensorWrapper)):
            return attr.to(device)
        elif isinstance(attr, dict):
            return {key: TensorWrapper._to(val, device) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _detach(attr):
        if isinstance(attr, (torch.Tensor, TensorWrapper)):
            return attr.detach()
        elif isinstance(attr, dict):
            return {key: TensorWrapper._detach(val) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _numpy(attr):
        if isinstance(attr, (torch.Tensor, TensorWrapper)):
            np_array = attr.numpy()
            if isinstance(np_array, np.ndarray):
                return np.ascontiguousarray(np_array, dtype=np_array.dtype)
            return np_array
        elif isinstance(attr, dict):
            return {key: TensorWrapper._numpy(val) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _torch(attr):
        if isinstance(attr, np.ndarray):
            return torch.tensor(attr)
        elif isinstance(attr, TensorWrapper):
            return attr.torch()
        elif isinstance(attr, dict):
            return {key: TensorWrapper._torch(val) for key, val in attr.items()}
        else:
            return attr

    @staticmethod
    def _eq(self_attr, other_attr):
        try:
            if isinstance(self_attr, torch.Tensor):
                return torch.allclose(self_attr, other_attr, equal_nan=True)
            elif isinstance(self_attr, dict):
                return all([TensorWrapper._eq(self_attr[k], other_attr[k]) for k in self_attr.keys()])
            else:
                return self_attr == other_attr
        except:  # noqa: E722
            return False

    def __getitem__(self, subscript):
        """By default, idiomatic slicing is used for the sequence dimension across batches.
        For batching use `slice_batch` instead.
        """
        return self.__class__(
            **{field.name: self._getitem(getattr(self, field.name), subscript) for field in fields(self.__class__)}
        )

    def slice_batch(self, subscript):
        return self.__class__(
            **{field.name: self._slice_batch(getattr(self, field.name), subscript) for field in fields(self.__class__)}
        )

    def select_active(self, mask: TensorType["batch_size", 1, int]):
        return self.__class__(
            **{field.name: self._select_active(getattr(self, field.name), mask) for field in fields(self.__class__)}
        )

    def to(self, device: str):
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            setattr(self, field.name, self._to(attr, device))
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def detach(self):
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            setattr(self, field.name, self._detach(attr))
        return self

    def numpy(self):
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            setattr(self, field.name, self._numpy(attr))
        return self

    def torch(self):
        for field, val in self.to_dict().items():
            setattr(self, field, self._torch(val))
        return self

    def clone(self):
        out_params = {}
        for field in fields(self.__class__):
            attr = getattr(self, field.name)
            if isinstance(attr, (torch.Tensor, TensorWrapper)):
                out_params[field.name] = attr.clone()
            elif attr is not None:
                out_params[field.name] = deepcopy(attr)
            else:
                out_params[field.name] = None
        return self.__class__(**out_params)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __str__(self):
        return f"{self.__class__.__name__}({pretty_dict(self.__dict__)})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        equals = {field: self._eq(val, getattr(other, field)) for field, val in self.__dict__.items()}
        return all(x for x in equals.values())

    def __json_encode__(self):
        return self.clone().detach().to("cpu").numpy().to_dict()

    def __json_decode__(self, **attrs):
        # Does not contemplate the usage of __slots__
        self.__dict__ = attrs
        self.__post_init__()

    def __post_init__(self):
        pass
