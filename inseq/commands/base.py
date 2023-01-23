import dataclasses
from abc import ABC, abstractstaticmethod
from argparse import Namespace
from typing import Any, Iterable, NewType, Union

from ..utils import InseqArgumentParser

DataClassType = NewType("DataClassType", Any)
OneOrMoreDataClasses = Union[DataClassType, Iterable[DataClassType]]


class BaseCLICommand(ABC):
    """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/commands/__init__.py"""

    _name: str = None
    _help: str = None
    _dataclasses: OneOrMoreDataClasses = None

    @classmethod
    def register_subcommand(cls, parser: InseqArgumentParser):
        """
        Register this command to argparse so it's available for the Inseq cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        command_parser = parser.add_parser(
            cls._name,
            help=cls._help,
            dataclass_types=cls._dataclasses,
        )
        command_parser.set_defaults(factory_method=cls.build)

    @classmethod
    def build(cls, args: Namespace):
        dataclasses_args = []
        if not isinstance(cls._dataclasses, tuple):
            cls._dataclasses = (cls._dataclasses,)
        for dataclass_type in cls._dataclasses:
            keys = {f.name for f in dataclasses.fields(dataclass_type) if f.init}
            inputs = {k: v for k, v in vars(args).items() if k in keys}
            dataclasses_args.append(dataclass_type(**inputs))
        if len(dataclasses_args) == 1:
            dataclasses_args = dataclasses_args[0]
        return cls, dataclasses_args

    @abstractstaticmethod
    def run(args: OneOrMoreDataClasses):
        raise NotImplementedError()
