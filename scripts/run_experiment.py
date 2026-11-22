"""Reproduce the paper's experiments: `python scripts/run_experiment.py [--config configs/default.yaml]`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upi_ids.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["experiment", *sys.argv[1:]])
