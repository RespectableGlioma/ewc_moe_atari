from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def is_colab() -> bool:
    """Best-effort Colab detection."""
    try:
        import google.colab  # type: ignore

        return True
    except Exception:
        return False


def mount_gdrive(mount_point: str = "/content/drive", *, force_remount: bool = False) -> None:
    """Mount Google Drive when running inside a Colab notebook.

    Notes
    - This requires user interaction (authorization) in the notebook UI.
    - In practice, it's usually better to mount in a notebook cell once:
        from google.colab import drive
        drive.mount('/content/drive')
    """
    try:
        from google.colab import drive  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mount_gdrive() only works in Google Colab") from e

    drive.mount(mount_point, force_remount=force_remount)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomic-ish torch.save.

    We write to a temporary file then rename.
    """
    import torch

    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def make_run_name(prefix: str = "run") -> str:
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S")


class JsonlLogger:
    """Simple JSONL metrics writer (line-buffered)."""

    def __init__(self, path: Path):
        ensure_dir(path.parent)
        self.path = path
        self._fh = path.open("a", buffering=1)

    def log(self, row: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(row) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
