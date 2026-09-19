"""Compatibility entrypoint; implementation lives in ``data_pipeline``."""

from data_pipeline.freeze_r7_split_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())
