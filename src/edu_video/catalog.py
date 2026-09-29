"""Explicit bridge to a dataset checkout; no catalog or pickle loading in the runner."""
import json
import subprocess
import sys
from pathlib import Path


class Catalog:
    def __init__(self, dataset_dir):
        self.root = Path(dataset_dir).expanduser().resolve()
        self.script = self.root / "scripts/plan_experiment.py"
        if not self.script.is_file():
            raise FileNotFoundError(f"Dataset checkout lacks scripts/plan_experiment.py: {self.root}")

    def _call(self, *args):
        result = subprocess.run([sys.executable, str(self.script), *args], cwd=self.root,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        if result.returncode:
            raise RuntimeError(f"Dataset planner failed: {result.stderr}")
        return json.loads(result.stdout)

    def search(self, query):
        return self._call("--query", query)

    def plan(self, experiment_id, *, durations_file=None):
        args = ["--experiment-id", experiment_id]
        if durations_file:
            args += ["--durations", str(Path(durations_file).resolve())]
        return self._call(*args)
