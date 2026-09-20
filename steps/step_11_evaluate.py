"""Шаг 11: оценка — сводный отчёт о прогрессе (текст + SVG-график loss)."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_cfg


def read_logs(cfg):
    steps = []
    for f in sorted(glob.glob(cfg.p(cfg.logs_dir, "step_*.jsonl"))):
        records = []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            steps.append(records)
    return steps


def build_svg(steps_by_log) -> str:
    pts = []
    for recs in steps_by_log:
        for r in recs:
            if r.get("event") == "train" and r.get("loss") is not None:
                pts.append((len(pts), float(r["loss"])))
    if not pts:
        return ""
    W, H, P = 900, 320, 50
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = 0, max(xs)
    ymin, ymax = min(ys) * 0.9, max(ys) * 1.05
    if ymax <= ymin:
        ymax = ymin + 1
    xspan = max(1, xmax - xmin)
    yspan = max(1e-9, ymax - ymin)

    def mx(x): return P + (x - xmin) / xspan * (W - 2 * P)
    def my(y): return H - P - (y - ymin) / yspan * (H - 2 * P)

    coords = [(mx(x), my(y)) for x, y in pts]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    dots = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#58a6ff"/>'
        for x, y in coords
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
<rect x="0" y="0" width="{W}" height="{H}" fill="#0d1117"/>
<text x="{P}" y="28" fill="#e6edf3" font-size="14" font-family="monospace">loss по шагам</text>
<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#30363d"/>
<line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#30363d"/>
<path d="{path}" fill="none" stroke="#58a6ff" stroke-width="2"/>
{dots}
<text x="{W-P}" y="{H-14}" fill="#8b949e" font-size="11" font-family="monospace" text-anchor="end">
итераций: {len(pts)} · loss {max(ys):.3f} → {min(ys):.3f}</text>
</svg>"""


def main() -> int:
    cfg = get_cfg()
    steps = read_logs(cfg)
    if not steps:
        print("[!] Нет журналов (logs/step_*.jsonl) — сначала запусти хотя бы один training-шаг.")
        return 1

    print("# Отчёт о прогрессе обучения\n")
    for recs in steps:
        start = next((r for r in recs if r.get("event") == "start"), {})
        train_events = [r for r in recs if r.get("event") == "train"]
        if train_events:
            print(f"- шаг {start.get('step')}: loss {train_events[0]['loss']:.4f} → "
                  f"{train_events[-1]['loss']:.4f}, {len(train_events)} точек, "
                  f"{train_events[-1].get('tokens_per_sec', 0):.0f} tok/s")

    svg = build_svg(steps)
    if svg:
        out = cfg.p(cfg.logs_dir, "progress.svg")
        with open(out, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"\n[svg] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
