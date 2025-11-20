# ForceCurveWidget

Interactive tool to explore configurable cable-tension force curves for a single repetition. The Streamlit UI wraps the `simulate_rep` helper so you can tweak concentric/eccentric timing, multipliers, and sensor effects, then inspect the resulting trace or export it for downstream analysis.

## Features

- Full control over every timing/multiplier parameter from the analytic model
- Adjustable sampling frequency
- Optional Gaussian noise, drift, and ADC-style quantization (bits + range)
- Plotly visualization with dual-axis option for ADC counts
- Instant CSV export and sortable raw data table

## Getting started

Requires Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app launches at `http://localhost:8501`. Open the sidebar to experiment with tempos or enable the sensor-effects expander to approximate your data-acquisition stack.

## Next ideas

- Batch-generate multiple reps with varying `W_kN` for synthetic datasets
- Save/load presets for common tempo prescriptions
- Overlay real reps vs. simulated reps for quick comparisons
