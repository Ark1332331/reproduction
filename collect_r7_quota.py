"""Compatibility entrypoint; implementation lives in ``simulation``."""

from simulation.collect_r7_quota import main


if __name__ == "__main__":
    raise SystemExit(main())
