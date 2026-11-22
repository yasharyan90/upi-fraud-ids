"""Stream synthetic UPI events through the (in-memory or Kafka) bus and the hybrid detector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upi_ids.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["simulate", *sys.argv[1:]])
