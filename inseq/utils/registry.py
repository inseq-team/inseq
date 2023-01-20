from __future__ import annotations

from abc import ABC
from typing import TypeVar

R = TypeVar("R", bound="Registry")


class Registry(ABC):
    registry_attr = "None"  # Override in child classes

    def __init__(self) -> None:
        if self.__class__ is Registry or self.__class__ in Registry.__subclasses__():
            raise OSError(
                f"{self.__class__.__name__} is designed to be instantiated "
                f"using the `{self.__class__.__name__}.load(name, **kwargs)` method."
            )

    @classmethod
    def subclasses(cls: type[R]) -> set[type[R]]:
        registry = set()
        for subclass in cls.__subclasses__():
            if subclass not in registry and subclass not in Registry.__subclasses__():
                registry.add(subclass)
            registry |= subclass.subclasses()
        return registry

    @classmethod
    def available_classes(cls: type[R]) -> dict[str, type[R]]:
        methods = {getattr(c, cls.registry_attr): c for c in cls.subclasses()}
        if cls is not Registry and cls not in Registry.__subclasses__():
            methods[getattr(cls, cls.registry_attr)] = cls
        return methods


def get_available_methods(cls: type[Registry]) -> list[str]:
    return list(cls.available_classes().keys())
