import subprocess
import sys


def test_scaling_requires_external_inputs(tmp_path):
    result = subprocess.run(
        [sys.executable, 'scripts/run_scaling.py', '--subset-dir', str(tmp_path)],
        text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Missing 20% Gowalla subset input' in result.stderr
    assert 'loc-gowalla_totalCheckins.txt' in result.stderr
