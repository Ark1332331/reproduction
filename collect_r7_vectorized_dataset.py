"""Compatibility entrypoint; implementation lives in ``simulation``."""

from simulation.collect_r7_vectorized_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
