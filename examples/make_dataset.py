"""Generate explicitly synthetic labeled data to exercise the NPZ contract.

This is a plumbing example, NOT a real accuracy dataset.
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic-labels.npz"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, inputs=rng.standard_normal((24, 16)).astype(np.float32),
             labels=rng.integers(0, 4, 24, dtype=np.int64))
    print(f"Synthetic labels (not real task accuracy): {args.output}")


if __name__ == "__main__":
    main()
