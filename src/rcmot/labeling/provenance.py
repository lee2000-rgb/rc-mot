"""Schema-v2 bookkeeping shared by the evaluation-side label tools (docs/13 v2, gt_design §C.2 / §D.6).

gt.json is read and written here as a plain dict (json.load / json.dump, indent 1, key order kept) so that keys the
tools do not know about are never dropped. `meta.provenance` is append-only: every tool that writes into gt.json
adds one {stage, who, tool, git, time, note} entry.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

GIT_NONE = "none"


def git_sha(root: str | Path) -> str:
    """Short git revision of the tool tree, or "none" (the project is not always a git checkout)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else GIT_NONE
    except (OSError, subprocess.SubprocessError):
        return GIT_NONE


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def load_doc(path: str | Path) -> dict:
    return json.load(open(path))


def save_doc(doc: dict, path: str | Path, backup: bool = True, keep: int = 30):
    """Write gt.json in place (backup/gt_<time>.json kept like GT.save, at most `keep` files)."""
    path = Path(path)
    if backup and path.exists():
        bdir = path.parent / "backup"
        bdir.mkdir(exist_ok=True)
        shutil.copy(path, bdir / f"gt_{time.strftime('%Y%m%d_%H%M%S')}.json")
        for f in sorted(bdir.glob("gt_*.json"))[:-keep]:
            f.unlink()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    tmp.replace(path)


def add_provenance(doc: dict, stage: str, who: str, tool: str, root: str | Path, note: str = "") -> dict:
    entry = {"stage": stage, "who": who, "tool": tool, "git": git_sha(root), "time": now_str(), "note": note}
    doc.setdefault("meta", {}).setdefault("provenance", []).append(entry)
    return entry


def ref3d_of(box: dict) -> dict | None:
    """The usable LiDAR reference of a GT box (None when absent or quality == none)."""
    r = box.get("ref3d")
    if not r or r.get("quality", "none") == "none" or r.get("range_m") is None:
        return None
    return r
