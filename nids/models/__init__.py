"""Model definitions and training utilities."""

from importlib import import_module as _imp
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING

__all__: list[str] = [
    "random_forest",
]

if TYPE_CHECKING:
    from . import random_forest  # noqa: F401
else:
    def __getattr__(name: str) -> _ModuleType:  # pragma: no cover
        if name in __all__:
            mod = _imp("%s.%s" % (__name__, name))
            globals()[name] = mod
            return mod
        raise AttributeError(name) 