"""Small, dependency-light provenance helpers for R7 capture files.

The paper does not publish the complete simulator/data-generation repository.
These fields therefore do not make the capture identical to the paper; they make
every local capture auditable and prevent an implementation assumption from
silently becoming an undocumented dataset property.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


def file_sha256(path: str | Path) -> str | None:
    """Return a file digest, or ``None`` when the external file is unavailable."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: str | Path) -> str | None:
    """Read the repository revision without making git state a hard dependency."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def installed_version(distribution: str) -> str | None:
    """Return an installed package version without importing simulator packages."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_provenance(
    *,
    collector_path: str | Path,
    checkpoint_path: str | Path,
    task: str,
    terrain_profile: str,
    map_size_m: float,
    voxel_size_m: float,
    grid_size: int,
    rollout_steps: int,
    camera_count: int,
    camera_tilt_degrees: float,
) -> dict:
    """Build JSON-safe provenance attached to each trajectory metadata record."""
    collector = Path(collector_path).resolve()
    repository = collector.parent
    return {
        "provenance_schema_version": 1,
        "collector": str(collector),
        "collector_sha256": file_sha256(collector),
        "repository_git_revision": git_revision(repository),
        "command": list(sys.argv),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            name: installed_version(name)
            for name in ("numpy", "torch", "isaaclab", "isaacsim", "trimesh")
        },
        "task": task,
        "policy_checkpoint": str(Path(checkpoint_path)),
        "policy_checkpoint_sha256": file_sha256(checkpoint_path),
        "terrain_profile": terrain_profile,
        "paper_contract": {
            "map_size_m": float(map_size_m),
            "voxel_size_m": float(voxel_size_m),
            "grid_size": int(grid_size),
            "rollout_steps": int(rollout_steps),
            "camera_count": int(camera_count),
            "camera_tilt_degrees": float(camera_tilt_degrees),
        },
    }
