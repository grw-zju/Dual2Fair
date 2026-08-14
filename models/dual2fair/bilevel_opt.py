import warnings

from .hierarchical_opt import HierarchicalAlternatingOptimizer

warnings.warn(
    'bilevel_opt is deprecated; use HierarchicalAlternatingOptimizer',
    DeprecationWarning, stacklevel=2)

BiLevelOptimizer = HierarchicalAlternatingOptimizer

__all__ = ['HierarchicalAlternatingOptimizer', 'BiLevelOptimizer']
