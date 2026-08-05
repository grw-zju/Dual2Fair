from .alternating import alternating_step
from .joint import joint_weighted_sum_step
from .mirror_alternating import MirrorAlternatingOptimizer
from .unrolled_bilevel import unrolled_bilevel_step

OPTIMIZATION_MODES = {
    'joint_weighted_sum': joint_weighted_sum_step,
    'alternating': alternating_step,
    'mirror_alternating': MirrorAlternatingOptimizer,
    'unrolled_bilevel': unrolled_bilevel_step,
}

__all__ = ['OPTIMIZATION_MODES', 'MirrorAlternatingOptimizer',
           'alternating_step', 'joint_weighted_sum_step',
           'unrolled_bilevel_step']
