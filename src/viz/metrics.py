from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}/{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def metrics_to_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    flat = [flatten_dict(r) for r in records]
    if not flat:
        return pd.DataFrame()
    df = pd.DataFrame(flat)
    # keep step if present
    if '_step' in df.columns:
        df = df.sort_values('_step')
    return df
