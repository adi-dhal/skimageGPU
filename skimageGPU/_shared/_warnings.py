import warnings
import functools
__all__ = ['warn']
warn = functools.partial(warnings.warn, stacklevel=2)
