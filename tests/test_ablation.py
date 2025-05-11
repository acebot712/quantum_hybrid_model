import subprocess

def test_ablation_runs():
    result = subprocess.run(['python', 'benchmarks/ablation.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"Ablation script failed: {result.stderr}" 