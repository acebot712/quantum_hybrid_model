import subprocess

def test_benchmark_runs():
    result = subprocess.run(['python', 'benchmarks/benchmark.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"Benchmark script failed: {result.stderr}" 