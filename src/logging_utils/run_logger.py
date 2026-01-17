from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


def _flatten(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    if isinstance(obj, (int, float)):
        out[prefix] = float(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
        return
    # ignore lists/strings for scalar logging


class RunLogger:
    """Writes JSONL + TensorBoard scalars into a run directory.

    This is designed for Colab:
    - JSONL survives even if TB isn't used
    - TensorBoard gives quick interactive plots
    """

    def __init__(self, run_dir: str | Path, *, config: Optional[object] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))

        if config is not None:
            cfg_path = self.run_dir / "config.json"
            try:
                payload = asdict(config)  # dataclass
            except Exception:
                payload = dict(config.__dict__)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

    def log(self, step: int, record: Dict[str, Any]) -> None:
        record = dict(record)
        record["_time"] = time.time()
        record["_step"] = int(step)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        scalars: Dict[str, float] = {}
        _flatten("", record, scalars)
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
