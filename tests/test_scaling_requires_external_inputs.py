import subprocess
import sys


def test_scaling_requires_external_inputs(tmp_path):
    result = subprocess.run(
        [sys.executable, 'scripts/benchmark_scaling.py', '--subset-dir', str(tmp_path)],
        text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Prepare exact external Gowalla subsets' in result.stderr
    assert 'does not invent subset manifests' in result.stderr
