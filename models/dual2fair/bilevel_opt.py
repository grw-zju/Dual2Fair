import warnings

from .hierarchical_opt import HierarchicalAlternatingOptimizer

warnings.warn(
    'models.dual2fair.bilevel_opt is deprecated. It is a compatibility shim, '
    'not a classical bi-level solver; use HierarchicalAlternatingOptimizer.',
    DeprecationWarning, stacklevel=2)

BiLevelOptimizer = HierarchicalAlternatingOptimizer

__all__ = [
    'BiLevelOptimizer',
    'HierarchicalAlternatingOptimizer',
]
