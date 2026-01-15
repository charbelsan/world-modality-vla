#!/usr/bin/env python3
"""
Parse LeRobot text logs (train/eval) into structured CSV and lightweight HTML plots.

Why this exists:
- We often run long jobs with `wandb.enable=false` and only keep stdout/stderr logs.
- We still want post-hoc diagnostics: loss curves, lr, grad norms, world gates, etc.

Works best with logs containing lines like:
  ... step:12K ... loss:0.070 grdn:0.215 lr:6.8e-05 updt_s:0.497 data_s:0.037
and/or additional metrics printed as `key:value` or `key=value`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


_KV_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_.-]*)"
    r"(?P<sep>[:=])"
    r"(?P<val>-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<suf>[KMG])?\b"
)

_RUN_MARKER_RE = re.compile(r"^===\s+(?P<kind>Train|Eval)\s+(?P<name>.+?)\s+seed=(?P<seed>\d+)\s+->\s+(?P<out>.+?)\s*===$")


def _parse_scaled_number(raw: str, suffix: str | None) -> float:
    x = float(raw)
    if not suffix:
        return x
    if suffix == "K":
        return x * 1e3
    if suffix == "M":
        return x * 1e6
    if suffix == "G":
        return x * 1e9
    return x


def _iter_kv_pairs(line: str) -> Iterable[Tuple[str, float]]:
    for m in _KV_RE.finditer(line):
        key = m.group("key")
        val = _parse_scaled_number(m.group("val"), m.group("suf"))
        yield key, val


@dataclass(frozen=True)
class RunKey:
    name: str
    seed: str
    out_dir: str
    kind: str  # "Train" or "Eval"

    @property
    def slug(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name.strip())
        return f"{safe}_seed{self.seed}"


def _guess_run_key_from_marker(line: str) -> Optional[RunKey]:
    m = _RUN_MARKER_RE.match(line.strip())
    if not m:
        return None
    return RunKey(
        name=m.group("name"),
        seed=m.group("seed"),
        out_dir=m.group("out"),
        kind=m.group("kind"),
    )


def parse_log_text(lines: Iterable[str]) -> Dict[RunKey, pd.DataFrame]:
    """
    Split a mixed log into runs using our launcher markers, and parse all numeric key/vals.

    If no markers are found, returns a single run called "log" with seed "0".
    """
    current: Optional[RunKey] = None
    rows: Dict[RunKey, List[Dict[str, Any]]] = {}
    any_marker = False

    for raw in lines:
        marker = _guess_run_key_from_marker(raw)
        if marker is not None:
            any_marker = True
            current = marker
            rows.setdefault(current, [])
            continue

        kvs = dict(_iter_kv_pairs(raw))
        if not kvs:
            continue

        if current is None and not any_marker:
            current = RunKey(name="log", seed="0", out_dir="", kind="Train")
            rows.setdefault(current, [])

        if current is None:
            # We found markers, but this line is before the first marker; ignore.
            continue

        kvs["_raw"] = raw.rstrip("\n")
        rows[current].append(kvs)

    out: Dict[RunKey, pd.DataFrame] = {}
    for rk, rs in rows.items():
        df = pd.DataFrame(rs)
        # Prefer `step` as x-axis if present.
        if "step" in df.columns:
            df = df.sort_values("step", kind="stable")
        out[rk] = df.reset_index(drop=True)
    return out


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _try_plotly_html(out_html: Path, df: pd.DataFrame, title: str, metrics: List[str]) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    x = None
    if "step" in df.columns:
        x = df["step"]
    elif "smpl" in df.columns:
        x = df["smpl"]
    else:
        x = list(range(len(df)))

    fig = go.Figure()
    for m in metrics:
        if m in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[m], mode="lines", name=m))

    fig.update_layout(
        title=title,
        xaxis_title="step" if "step" in df.columns else ("samples" if "smpl" in df.columns else "index"),
        yaxis_title="value",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=520,
    )
    out_html.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to a LeRobot stdout/stderr log file.")
    ap.add_argument("--out", required=True, help="Output directory for CSV/HTML summaries.")
    ap.add_argument(
        "--metrics",
        default="loss,loss_total,world_loss,world_gate,lr,grdn,updt_s,data_s,world_cos",
        help="Comma-separated metric names to plot when present.",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    _safe_mkdir(out_dir)

    parsed = parse_log_text(log_path.read_text(errors="replace").splitlines(True))

    manifest: Dict[str, Any] = {
        "log": str(log_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "runs": [],
    }

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    for rk, df in parsed.items():
        run_dir = out_dir / rk.slug
        _safe_mkdir(run_dir)

        csv_path = run_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)

        # Basic stats.
        stats: Dict[str, Any] = {"rows": int(len(df))}
        for m in metrics:
            if m in df.columns and len(df[m].dropna()) > 0:
                series = pd.to_numeric(df[m], errors="coerce").dropna()
                if len(series) > 0:
                    stats[m] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "last": float(series.iloc[-1]),
                    }
        _write_json(run_dir / "stats.json", stats)

        title = f"{rk.kind} {rk.name} (seed={rk.seed})"
        plotted = _try_plotly_html(run_dir / "plot.html", df, title=title, metrics=metrics)

        manifest["runs"].append(
            {
                "kind": rk.kind,
                "name": rk.name,
                "seed": rk.seed,
                "out_dir": rk.out_dir,
                "csv": str(csv_path),
                "plot_html": str(run_dir / "plot.html") if plotted else None,
            }
        )

    _write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote {len(parsed)} run(s) to: {out_dir}")


if __name__ == "__main__":
    main()

