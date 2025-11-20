from __future__ import annotations

from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from force_curve_widget.simulate import (
    NoiseSettings,
    SimulationParams,
    simulate_rep,
    simulate_rep_simple,
)

SIMPLE_DEFAULT_WEIGHT = 80.0
SIMPLE_DEFAULT_CONCENTRIC = 1.0
SIMPLE_DEFAULT_PAUSE = 0.2
SIMPLE_DEFAULT_ECCENTRIC = 2.0
SIMPLE_DEFAULT_STROKE = 0.5
KILONEWTON_TO_LBF = 224.80894387096

st.set_page_config(page_title="Force Curve Widget", layout="wide", page_icon="🏋️")
st.title("Force Curve Simulator")
st.caption(
    "This tool simulates the force curve of a weight stack during a lift.\n"
    "Variables include concentric and eccentric durations, pause times, and more.\n\n"
    "Developed by Matt Gross."
)



def sidebar_inputs() -> tuple[bool, SimulationParams | dict[str, float], NoiseSettings, int | None]:
    with st.sidebar:
        st.header("Control mode")
        simple_mode = st.toggle(
            "Simple controls",
            value=True,
            help="Switch between high-level inputs and per-phase tempo parameters.",
        )

        if simple_mode:
            st.subheader("Simple inputs")
            weight_unit = st.radio("Units", options=("lbs", "kg"), horizontal=True, index=0)
            weight = st.number_input(
                "Stack weight",
                min_value=5.0,
                max_value=500.0,
                value=SIMPLE_DEFAULT_WEIGHT,
                step=5.0,
            )
            t_con = st.slider(
                "Concentric duration (s)", 0.2, 3.0, SIMPLE_DEFAULT_CONCENTRIC, 0.05, key="simple_conc"
            )
            t_pause = st.slider(
                "Pause at top (s)", 0.0, 1.5, SIMPLE_DEFAULT_PAUSE, 0.05, key="simple_pause"
            )
            t_ecc = st.slider(
                "Eccentric duration (s)", 0.3, 5.0, SIMPLE_DEFAULT_ECCENTRIC, 0.05, key="simple_ecc"
            )
            fs = st.slider("Sample rate (Hz)", 100, 2000, 500, 50, key="simple_fs")
            stroke_m = st.number_input(
                "Stack travel (m)", min_value=0.1, max_value=1.0, value=SIMPLE_DEFAULT_STROKE, step=0.05
            )
            payload: SimulationParams | dict[str, float] = {
                "weight_value": weight,
                "weight_unit": weight_unit,
                "t_concentric": t_con,
                "t_pause_top": t_pause,
                "t_eccentric": t_ecc,
                "fs": fs,
                "stroke_m": stroke_m,
            }
        else:
            st.subheader("Concentric")
            c1 = st.number_input("Accel up t_c1 (s)", min_value=0.05, max_value=1.0, value=0.30, step=0.05)
            c2 = st.number_input("Steady up t_c2 (s)", min_value=0.05, max_value=1.5, value=0.40, step=0.05)
            c3 = st.number_input("Decel t_c3 (s)", min_value=0.05, max_value=1.0, value=0.20, step=0.05)

            st.subheader("Isometric & Eccentric")
            t_top = st.number_input("Top pause (s)", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
            e1 = st.number_input("Early eccentric t_e1 (s)", min_value=0.05, max_value=1.5, value=0.40, step=0.05)
            e2 = st.number_input("Braking eccentric t_e2 (s)", min_value=0.05, max_value=2.0, value=0.60, step=0.05)

            st.subheader("Force multipliers")
            W_kN = st.number_input("Hold force W_kN", min_value=0.05, max_value=5.0, value=0.36, step=0.01)
            k_spike = st.slider("Concentric spike k_spike", 1.0, 4.0, 1.5, 0.05)
            k_plateau = st.slider("Plateau k_plateau", 0.8, 2.0, 1.05, 0.01)
            k_low = st.slider("Low eccentric k_low", 0.0, 1.0, 0.25, 0.01)
            k_brake = st.slider("Braking peak k_brake", 1.0, 3.0, 1.25, 0.05)

            fs = st.slider("Sample rate (Hz)", 100, 2000, 500, 50, key="advanced_fs")
            stroke_m = st.number_input(
                "Cable travel (m)", min_value=0.1, max_value=1.0, value=SIMPLE_DEFAULT_STROKE, step=0.05
            )
            payload = SimulationParams(
                W_kN=W_kN,
                t_c1=c1,
                t_c2=c2,
                t_c3=c3,
                t_top=t_top,
                t_e1=e1,
                t_e2=e2,
                k_spike=k_spike,
                k_plateau=k_plateau,
                k_low_ecc=k_low,
                k_brake=k_brake,
                fs=fs,
                stroke_m=stroke_m,
            )

        seed_opt = st.text_input("Random seed (optional)", value="")
        seed = int(seed_opt) if seed_opt.strip().isdigit() else None

        with st.expander("Sensor effects", expanded=False):
            noise_enabled = st.checkbox("Add Gaussian noise", value=False)
            noise_std = st.number_input(
                "Noise σ (kN)", value=0.005, min_value=0.0, max_value=0.2, step=0.001, disabled=not noise_enabled
            )
            drift = st.number_input(
                "Drift per second (kN/s)", value=0.0, min_value=-0.1, max_value=0.1, step=0.001, disabled=not noise_enabled
            )
            quant_enabled = st.checkbox("Quantize to ADC counts", value=False)
            bits = st.slider("ADC bits", min_value=8, max_value=24, value=16, disabled=not quant_enabled)
            full_scale = st.number_input(
                "Full-scale range (kN)", value=2.0, min_value=0.5, max_value=10.0, step=0.1, disabled=not quant_enabled
            )

        noise = NoiseSettings(
            noise_enabled=noise_enabled,
            noise_std=noise_std,
            drift_per_second=drift if noise_enabled else 0.0,
            quantization_enabled=quant_enabled,
            quantization_bits=bits,
            quantization_range_kN=full_scale,
        )

    return simple_mode, payload, noise, seed


def build_chart(
    df: pd.DataFrame,
    value_column: str,
    label: str,
    *,
    show_velocity_heatmap: bool = False,
    velocity_column: str | None = None,
    cursor_time: float | None = None,
    cursor_annotation: str | None = None,
    cursor_y: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df[value_column], name=label, mode="lines", line=dict(width=3))
    )

    if show_velocity_heatmap and velocity_column and velocity_column in df.columns:
        downsample = max(1, len(df) // 800)
        sample = df.iloc[::downsample].copy()
        if sample.index[-1] != df.index[-1]:
            sample = pd.concat([sample, df.iloc[[-1]]])
        sample = sample.sort_index()
        fig.add_trace(
            go.Scatter(
                x=sample["time_s"],
                y=sample[value_column],
                mode="markers",
                name="Velocity",
                marker=dict(
                    size=6,
                    color=sample[velocity_column],
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title="Velocity (m/s)",
                        y=0.5,
                        len=0.7,
                        thickness=12,
                        x=1.15,
                    ),
                    line=dict(width=0),
                ),
                hovertemplate="t=%{x:.2f}s<br>Force=%{y:.3f}<br>Vel=%{marker.color:.2f} m/s",
            )
        )

    if cursor_time is not None:
        fig.add_vline(x=cursor_time, line_color="#6c6c6c", line_dash="dot", line_width=2)
        if cursor_annotation is not None:
            fig.add_annotation(
                x=cursor_time,
                y=cursor_y if cursor_y is not None else df[value_column].max(),
                text=cursor_annotation,
                showarrow=False,
                bgcolor="rgba(0,0,0,0.6)",
                font=dict(color="white", size=12),
                yshift=35,
                borderpad=6,
                bordercolor="#999",
                borderwidth=1,
                align="left",
            )

    if "adc_counts" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df["adc_counts"],
                name="ADC counts",
                mode="lines",
                line=dict(width=1, dash="dot"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis=dict(title=label),
            yaxis2=dict(title="ADC counts", overlaying="y", side="right"),
        )
    else:
        fig.update_layout(yaxis=dict(title=label))

    fig.update_layout(
        xaxis_title="Time (s)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=10, b=40),
        height=420,
    )
    return fig


simple_mode, payload, noise, seed = sidebar_inputs()

if simple_mode:
    df = simulate_rep_simple(**payload, noise=noise, seed=seed)
    fs_current = payload["fs"]
else:
    params = cast(SimulationParams, payload)
    df = simulate_rep(params, noise=noise, seed=seed)
    fs_current = params.fs

rep_duration = float(df["time_s"].iloc[-1])


df["force_lbf"] = df["force_kN"] * KILONEWTON_TO_LBF
force_unit = st.radio(
    "Force display units",
    options=("Force (kN)", "Perceived weight (lbf)"),
    horizontal=True,
)

curve_style = st.radio(
    "Curve styling",
    options=("Solid force trace", "Velocity heatmap overlay"),
    horizontal=True,
)
show_heatmap = curve_style == "Velocity heatmap overlay"

if force_unit == "Force (kN)":
    display_column = "force_kN"
    display_label = "Force (kN)"
    metric_suffix = "kN"
    peak_label = "Peak force"
else:
    display_column = "force_lbf"
    display_label = "Perceived weight (lbf)"
    metric_suffix = "lbf"
    peak_label = "Peak perceived weight"

peak_force = float(df[display_column].max())
time_to_peak = float(df.loc[df[display_column].idxmax(), "time_s"])
max_con_vel = float(df["velocity_mps"].max())
max_ecc_vel = float(df["velocity_mps"].min())

cols = st.columns(3)
cols[0].metric(peak_label, f"{peak_force:.3f} {metric_suffix}")
cols[1].metric("Time to peak", f"{time_to_peak:.3f} s")
cols[2].metric("Rep duration", f"{rep_duration:.2f} s")

vel_cols = st.columns(2)
vel_cols[0].metric("Peak concentric velocity", f"{max_con_vel:.2f} m/s")
vel_cols[1].metric(
    "Peak eccentric velocity", f"{abs(max_ecc_vel):.2f} m/s", help="Absolute value during descent"
)

slider_max = max(rep_duration, 0.1)
cursor_step = max(round(1.0 / fs_current, 4), 0.001)
cursor_default = min(rep_duration, slider_max / 2)
cursor_time = st.slider(
    "Timeline cursor",
    min_value=0.0,
    max_value=slider_max,
    value=cursor_default,
    step=cursor_step,
    help="Drag to sweep a vertical bar across the rep and read values at that instant.",
)
cursor_time = min(cursor_time, df["time_s"].iloc[-1])
cursor_idx = int((df["time_s"] - cursor_time).abs().idxmin())
cursor_row = df.loc[cursor_idx]

cursor_force = float(cursor_row[display_column])
cursor_vel = float(cursor_row["velocity_mps"])
cursor_pos = float(cursor_row["position_m"])
cursor_annotation = (
    f"t={cursor_row['time_s']:.2f}s<br>{display_label}: {cursor_force:.3f} {metric_suffix}"
    f"<br>Velocity: {cursor_vel:.2f} m/s"
)

st.caption(
    f"Cursor → Force {cursor_force:.3f} {metric_suffix} · Velocity {cursor_vel:.2f} m/s · Position {cursor_pos:.2f} m"
)

st.plotly_chart(
    build_chart(
        df,
        display_column,
        display_label,
        show_velocity_heatmap=show_heatmap,
        velocity_column="velocity_mps",
        cursor_time=cursor_row["time_s"],
        cursor_annotation=cursor_annotation,
        cursor_y=cursor_force,
    ),
    width="stretch",
)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="force_curve.csv", mime="text/csv")

with st.expander("Raw data"):
    st.dataframe(df, width="stretch", height=300)
