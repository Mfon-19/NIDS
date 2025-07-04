"""Subpackage containing model training and inference pipelines."""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING, Any as _Any

__all__: list[str] = [
    "train",
    "predict",
]

if TYPE_CHECKING:
    from . import train, predict  # noqa: F401
else:
    def __getattr__(name: str) -> _ModuleType:  # pragma: no cover
        if name in __all__:
            module = _import_module(f"{__name__}.{name}")
            globals()[name] = module  # cache
            return module
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 