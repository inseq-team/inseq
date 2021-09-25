from typing import Dict, List, MutableSet, NoReturn, TypeVar

from abc import ABC


class Registry(ABC):
    attr = None  # Override in child classes

    def __init__(self) -> NoReturn:
        if self.__class__ is Registry or self.__class__ in Registry.__subclasses__():
            raise OSError(
                f"{self.__class__.__name__} is designed to be instantiated "
                f"using the `{self.__class__.__name__}.load(name, **kwargs)` method."
            )

    @classmethod
    def subclasses(cls) -> MutableSet:
        registry = set()
        for subclass in cls.__subclasses__():
            if subclass not in registry and subclass not in Registry.__subclasses__():
                registry.add(subclass)
            registry |= subclass.subclasses()
        return registry

    @classmethod
    def available_classes(cls) -> Dict[str, type]:
        methods = {getattr(c, cls.attr): c for c in cls.subclasses()}
        if cls is not Registry and cls not in Registry.__subclasses__():
            methods[getattr(cls, cls.attr)] = cls
        return methods


def get_available_methods(cls: TypeVar) -> List[str]:
    return [n for n in cls.available_classes().keys()]
