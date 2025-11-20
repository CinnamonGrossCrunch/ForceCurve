"""Quick smoke test for the simple-mode wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from force_curve_widget.simulate import simulate_rep_simple


if __name__ == "__main__":
    df = simulate_rep_simple()
    print(df.head())
    print(df.tail())
