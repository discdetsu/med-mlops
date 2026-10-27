"""DVC and git helpers. DVC runs as `python -m dvc` from this interpreter, so the flows work
the same under `uv run`, a Prefect worker, or CI without relying on PATH."""

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


class DatasetChangedError(RuntimeError):
    """The workspace doesn't match the DVC-pinned version of a dataset path."""


def dvc(root: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "dvc", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc {' '.join(args)} failed:\n{result.stderr or result.stdout}")
    return result.stdout


def dvc_add(root: Path, targets: Sequence[Path]) -> None:
    dvc(root, "add", *(str(t.relative_to(root)) for t in targets))


def dvc_push(root: Path, targets: Sequence[Path]) -> None:
    dvc(root, "push", *(str(dvc_file(t).relative_to(root)) for t in targets))


def dvc_pull(root: Path, targets: Sequence[Path]) -> None:
    dvc(root, "pull", *(str(dvc_file(t).relative_to(root)) for t in targets))


def is_clean(root: Path, target: Path) -> bool:
    """True when `target` exists and matches its .dvc file exactly."""
    if not target.exists():
        return False
    status = json.loads(dvc(root, "status", "--json", str(dvc_file(target).relative_to(root))))
    return not status


def dvc_file(target: Path) -> Path:
    return target.with_name(target.name + ".dvc")


def dvc_md5(target: Path) -> str:
    """The content hash pinned in `<target>.dvc`, e.g. `3f2a…c1.dir` for a directory."""
    outs = yaml.safe_load(dvc_file(target).read_text())["outs"]
    if len(outs) != 1:
        raise ValueError(f"{dvc_file(target)}: expected one output, got {len(outs)}")
    return str(outs[0]["md5"])


def dataset_rev(md5s: Mapping[str, str]) -> str:
    """One short ID for a set of pinned DVC paths: sha256 over sorted `path:md5` lines.

    Content-addressed, so the same data always gets the same rev whatever git commit or
    machine it was produced on. The per-path md5s are stored next to it.
    """
    lines = "\n".join(f"{path}:{md5}" for path, md5 in sorted(md5s.items()))
    return hashlib.sha256(lines.encode()).hexdigest()[:12]


def git_commit(root: Path) -> str:
    """HEAD sha, suffixed with `-dirty` when tracked files have uncommitted changes."""
    sha = _git(root, "rev-parse", "HEAD").strip()
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").strip()
    return f"{sha}-dirty" if dirty else sha


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout
