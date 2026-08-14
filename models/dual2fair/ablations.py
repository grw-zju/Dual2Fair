AVAILABLE_ABLATIONS = {
    'without_urc': {
        'label': 'w/o URC',
        'config': {'dual2fair.enable_user_calibration': False},
        'status': 'available',
    },
    'without_irc': {
        'label': 'w/o IRC',
        'config': {'dual2fair.enable_item_calibration': False},
        'status': 'available',
    },
    'uniform_target': {
        'label': 'Uniform Target',
        'config': {'dual2fair.omega': 1.0},
        'status': 'available',
    },
    'merit_only': {
        'label': 'Merit Only',
        'config': {'dual2fair.omega': 0.0},
        'status': 'available',
    },
    'without_confidence': {
        'label': 'w/o Confidence',
        'config': {'dual2fair.enable_confidence': False},
        'status': 'available',
    },
    'joint_weighted_sum': {
        'label': 'Joint Weighted Sum',
        'config': {'dual2fair.optimization_strategy': 'joint_weighted_sum'},
        'status': 'available',
    },
    'standard_alternating': {
        'label': 'Standard Alternating',
        'config': {'dual2fair.optimization_strategy': 'standard_alternating'},
        'status': 'available',
    },
    'hard_matching': {
        'label': 'Hard Matching',
        'config': {'dual2fair.alignment_mode': 'hard'},
        'status': 'available',
        'historical_implementation_found': False,
    },
    'mmd_alignment': {
        'label': 'MMD Alignment',
        'config': {'dual2fair.alignment_mode': 'mmd'},
        'status': 'available',
        'mmd_kernel': 'linear',
        'historical_implementation_found': False,
    },
    'dual2fair': {
        'label': 'Dual2Fair',
        'config': {'dual2fair.optimization_strategy': 'hierarchical_mirror'},
        'status': 'available',
    },
}


def get_ablation(name):
    if name in AVAILABLE_ABLATIONS:
        return AVAILABLE_ABLATIONS[name]
    raise KeyError(name)
