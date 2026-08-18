import pytest

from scripts.run_efficiency import benchmark


def test_standard_and_lowrank_benchmarks_run_on_demo():
    standard = benchmark('demo', 'lightgcn', '', 'standard', -1,
                         warmup_repeats=1, measure_repeats=1)
    lowrank = benchmark('demo', 'lightgcn', '', 'dual2fair_lowrank', -1,
                        warmup_repeats=1, measure_repeats=1)
    for result in (standard, lowrank):
        assert result['time_per_epoch_seconds'] >= 0
        assert result['peak_memory_bytes'] > 0
        assert result['inference_seconds_per_user'] >= 0
        assert result['dataset'] == 'demo'


def test_dense_benchmark_runs_on_demo_if_safe():
    dense = benchmark('demo', 'lightgcn', '', 'dual2fair_dense', -1,
                      warmup_repeats=1, measure_repeats=1)
    assert dense['calibration_refresh_seconds'] >= 0
