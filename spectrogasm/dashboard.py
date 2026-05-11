"""Self-contained interactive HTML dashboard.

Reads every artifact under ``results/<date>/`` produced by the calibration
pipeline (``solution.json``, ``rv.json``, ``science.json``, ``rv_lines.csv``,
``lineas_calibracion.csv``, ``thar_calibrado.csv``, ``estrella_calibrada.csv``)
and writes a single ``dashboard.html`` with:

* a summary header (palette-themed metric cards),
* calibration-quality plots (RMS, polynomial coefficients vs. date),
* radial-velocity panel (per-night v_rad with SEM + per-element comparison),
* science extras (spectral resolution, SNR, equivalent widths),
* a per-night drill-down with the Th-Ar lamp, the stellar spectrum and
  every Doppler line fit, all interactive (zoom, hover, toggle traces).

Run as::

    python -m spectrogasm.dashboard
    python -m spectrogasm.dashboard results/ docs/dashboard.html

The HTML loads Plotly from a CDN, so the file is tiny and viewable
offline once cached.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from . import style as _style  # noqa: F401  (ensures palette loaded)
from .style import DARK, INK, LIGHT, MUTED, PALETTE, PRIMARY, SECONDARY
from .manifest import RESULTS_DIR


# --- Plotly theme ---------------------------------------------------------

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="DejaVu Sans, Helvetica, Arial", size=13, color=INK),
        colorway=PALETTE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(font=dict(size=18, color=INK), x=0.02, xanchor="left"),
        xaxis=dict(showgrid=True, gridcolor="#ebd6df", zeroline=False,
                   linecolor=INK, ticks="outside", tickcolor=INK),
        yaxis=dict(showgrid=True, gridcolor="#ebd6df", zeroline=False,
                   linecolor=INK, ticks="outside", tickcolor=INK),
        legend=dict(bgcolor="white", bordercolor="#ebd6df", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=50),
        hoverlabel=dict(bgcolor=PRIMARY, font=dict(color="white")),
    )
)
pio.templates["spectrogasm"] = PLOTLY_TEMPLATE
pio.templates.default = "spectrogasm"


# --- Data loading --------------------------------------------------------

@dataclass
class NightData:
    date: str
    solution: dict
    rv: dict
    science: dict
    rv_lines: pd.DataFrame
    lines_used: pd.DataFrame
    thar: pd.DataFrame | None
    star: pd.DataFrame | None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def collect(results_dir: Path) -> list[NightData]:
    """Load every night's artifact set, sorted by date."""
    nights: list[NightData] = []
    for night_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        sol = _read_json(night_dir / "solution.json")
        if not sol:
            continue
        rv = _read_json(night_dir / "rv.json")
        sci = _read_json(night_dir / "science.json")
        rv_lines = _read_csv(night_dir / "rv_lines.csv")
        lines_used = _read_csv(night_dir / "lineas_calibracion.csv")
        thar = _read_csv(night_dir / "thar_calibrado.csv")
        star = _read_csv(night_dir / "estrella_calibrada.csv")
        nights.append(NightData(
            date=sol.get("date", night_dir.name),
            solution=sol, rv=rv, science=sci,
            rv_lines=rv_lines, lines_used=lines_used,
            thar=thar if not thar.empty else None,
            star=star if not star.empty else None,
        ))
    return nights


# --- Aggregated tables --------------------------------------------------

def overview_table(nights: list[NightData]) -> pd.DataFrame:
    rows = []
    for n in nights:
        rows.append({
            "date": n.date,
            "degree": n.solution.get("degree"),
            "rms_A": n.solution.get("rms"),
            "n_lines": n.solution.get("n_lines"),
            "seed": n.solution.get("seed_origin"),
            "v_tel_kms": n.solution.get("telluric_kms"),
            "v_rad_kms": n.rv.get("v_mean_kms"),
            "v_rad_sem": n.rv.get("v_sem_kms"),
            "R_median": (n.science.get("resolution") or {}).get("R_median"),
            "snr": n.science.get("snr"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date_dt"] = pd.to_datetime(df["date"])
        df = df.sort_values("date_dt").reset_index(drop=True)
    return df


# --- Figure builders ----------------------------------------------------

def _fig_to_div(fig: go.Figure, div_id: str) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       div_id=div_id, default_height="420px")


def fig_rms(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date_dt"], y=df["rms_A"],
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=10, color=SECONDARY,
                    line=dict(color=PRIMARY, width=1.5)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>RMS = %{y:.4f} Å<extra></extra>",
        name="RMS",
    ))
    fig.update_layout(title="Calidad de la solución λ — RMS por noche",
                      yaxis_title="RMS (Å)", xaxis_title="Fecha")
    return fig


def fig_coefficients(nights: list[NightData]) -> go.Figure:
    """One subplot per polynomial coefficient, stacked vertically."""
    from plotly.subplots import make_subplots

    rows = []
    for n in nights:
        coef = n.solution.get("coef", [])
        rows.append({"date": pd.Timestamp(n.date),
                     **{f"c{len(coef)-1-i}": c for i, c in enumerate(coef)}})
    df = pd.DataFrame(rows).sort_values("date")
    coef_cols = sorted((c for c in df.columns if c.startswith("c")),
                       key=lambda c: -int(c[1:]))
    n_panels = len(coef_cols)

    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=coef_cols)
    for i, col in enumerate(coef_cols, start=1):
        color = PALETTE[(i - 1) % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8, color=SECONDARY,
                        line=dict(color=color, width=1.2)),
            name=col, showlegend=False,
        ), row=i, col=1)
        fig.update_yaxes(title_text=col, row=i, col=1)
    fig.update_xaxes(title_text="Fecha", row=n_panels, col=1)
    fig.update_layout(title="Coeficientes del polinomio λ(pix) vs fecha",
                      height=180 * n_panels + 60)
    return fig


def fig_n_lines(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["date_dt"], y=df["n_lines"],
        marker=dict(color=SECONDARY, line=dict(color=PRIMARY, width=1.2)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y} líneas<extra></extra>",
    ))
    fig.update_layout(title="Número de líneas usadas en el ajuste",
                      yaxis_title="N líneas", xaxis_title="Fecha")
    return fig


def fig_rv(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date_dt"], y=df["v_rad_kms"],
        error_y=dict(type="data", array=df["v_rad_sem"], color=PRIMARY,
                     thickness=1.6, width=6),
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=11, color=SECONDARY,
                    line=dict(color=PRIMARY, width=1.5)),
        hovertemplate=("<b>%{x|%Y-%m-%d}</b><br>v = %{y:+.3f} km/s"
                       "<br>SEM = %{error_y.array:.3f}<extra></extra>"),
        name="v̄ ± SEM",
    ))
    fig.update_layout(title="Velocidad radial estelar por noche · V/Ni/Ti",
                      yaxis_title="v_rad (km/s)", xaxis_title="Fecha")
    return fig


def fig_per_element(nights: list[NightData]) -> go.Figure:
    rows = []
    for n in nights:
        for r in n.science.get("per_element_rv", []) or []:
            rows.append({"date": pd.Timestamp(n.date), **r})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Velocidad por elemento — sin datos")
        return fig

    elem_colors = {"V": PRIMARY, "Ni": SECONDARY, "Ti": DARK}
    for elem, sub in df.groupby("element"):
        sub = sub.sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["v_mean_kms"],
            error_y=dict(type="data", array=sub["v_sem_kms"],
                         color=elem_colors.get(elem, MUTED),
                         thickness=1.4, width=6),
            mode="lines+markers", name=elem,
            line=dict(color=elem_colors.get(elem, MUTED), width=2),
            marker=dict(size=10,
                        color=elem_colors.get(elem, MUTED),
                        line=dict(color="white", width=1.2)),
        ))
    fig.update_layout(title="Velocidad radial por elemento (consistencia interna)",
                      yaxis_title="v (km/s)", xaxis_title="Fecha")
    return fig


def fig_resolution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["date_dt"], y=df["R_median"],
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=10, color=SECONDARY,
                    line=dict(color=PRIMARY, width=1.5)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>R = %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(title="Poder resolutivo del espectrógrafo R = λ/FWHM",
                      yaxis_title="R (mediana, por noche)",
                      xaxis_title="Fecha")
    return fig


def fig_snr(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["date_dt"], y=df["snr"],
        marker=dict(color=PALETTE[2], line=dict(color=PRIMARY, width=1.2)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>SNR ≈ %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(title="SNR del espectro estelar",
                      yaxis_title="SNR (mediana / σ robusto)",
                      xaxis_title="Fecha")
    return fig


def fig_ew_heatmap(nights: list[NightData]) -> go.Figure:
    rows = []
    for n in nights:
        for r in n.science.get("equivalent_widths", []) or []:
            if not r.get("ok"):
                continue
            rows.append({"date": pd.Timestamp(n.date),
                         "label": f"{r['element']} {r['lambda_lab']:.3f}",
                         "ew": r.get("ew_mA")})
    df = pd.DataFrame(rows)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Anchos equivalentes (mÅ) — sin datos")
        return fig
    pivot = df.pivot(index="label", columns="date", values="ew").sort_index()
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(d.date()) for d in pivot.columns],
        y=pivot.index, colorscale=[[0, LIGHT], [0.5, SECONDARY], [1.0, PRIMARY]],
        colorbar=dict(title="EW (mÅ)", outlinecolor="white"),
        hovertemplate="%{y}<br>%{x}<br>EW = %{z:.1f} mÅ<extra></extra>",
    ))
    fig.update_layout(title="Ancho equivalente por línea y noche",
                      xaxis_title="Fecha", yaxis_title="Línea (Å)")
    return fig


def fig_spectrum(spec: pd.DataFrame, title: str,
                 lambda_col: str = "lambda_cal", fill: bool = False,
                 lines_overlay: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if fill:
        fig.add_trace(go.Scatter(
            x=spec[lambda_col], y=spec["intensidad"],
            fill="tozeroy", mode="none",
            fillcolor="rgba(255,194,218,0.55)",
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=spec[lambda_col], y=spec["intensidad"],
        mode="lines", line=dict(color=PRIMARY, width=1.2),
        name="espectro",
        hovertemplate="λ = %{x:.3f} Å<br>I = %{y:.2f}<extra></extra>",
    ))
    if lines_overlay is not None and not lines_overlay.empty:
        ys = np.interp(lines_overlay["lambda_atlas"], spec[lambda_col],
                       spec["intensidad"])
        fig.add_trace(go.Scatter(
            x=lines_overlay["lambda_atlas"], y=ys,
            mode="markers",
            marker=dict(symbol="circle", size=10,
                        color=SECONDARY,
                        line=dict(color=PRIMARY, width=1.5)),
            name="líneas usadas",
            hovertemplate="λ_atlas = %{x:.3f} Å<extra></extra>",
        ))
    fig.update_layout(title=title, xaxis_title="Longitud de onda (Å)",
                      yaxis_title="Intensidad", height=420)
    return fig


def fig_rv_fits(night: NightData) -> go.Figure | None:
    """One small subplot per stellar line: data + λ_lab + measured λ_obs."""
    if night.star is None or night.rv_lines.empty:
        return None

    lam = night.star["lambda_rest"].to_numpy() \
        if "lambda_rest" in night.star.columns \
        else night.star["lambda_cal"].to_numpy()
    intensity = night.star["intensidad"].to_numpy()

    n = len(night.rv_lines)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=nrows, cols=ncols, horizontal_spacing=0.06, vertical_spacing=0.16,
        subplot_titles=[
            f"{r['element']} {r['lambda_lab']:.3f}  ·  "
            f"Δλ = {(r['lambda_obs'] - r['lambda_lab']):+.3f} Å  ·  "
            f"v = {r['v_kms']:+.2f} km/s"
            if r["ok"] else
            f"{r['element']} {r['lambda_lab']:.3f}  ·  (sin datos)"
            for _, r in night.rv_lines.iterrows()
        ],
    )
    half = 2.0
    for i, (_, row) in enumerate(night.rv_lines.iterrows()):
        rr, cc = i // ncols + 1, i % ncols + 1
        lam_lab = row["lambda_lab"]
        mask = (lam >= lam_lab - half) & (lam <= lam_lab + half)
        x = lam[mask]
        y = intensity[mask]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=DARK, width=1.2),
            marker=dict(size=6, color=DARK, opacity=0.8),
            showlegend=False,
            hovertemplate="λ=%{x:.3f}<br>I=%{y:.2f}<extra></extra>",
        ), row=rr, col=cc)
        if row["ok"] and np.isfinite(row.get("continuum", np.nan)):
            fig.add_hline(y=row["continuum"], line=dict(color=LIGHT, dash="dot", width=1),
                          row=rr, col=cc)
        if row["ok"]:
            fig.add_vline(x=row["lambda_obs"],
                          line=dict(color=PRIMARY, width=1.8),
                          row=rr, col=cc)
        fig.add_vline(x=lam_lab,
                      line=dict(color=SECONDARY, dash="dash", width=1.2),
                      row=rr, col=cc)
        fig.update_xaxes(title_text="λ (Å)", row=rr, col=cc)
        fig.update_yaxes(title_text="I", row=rr, col=cc)
    fig.update_layout(
        height=260 * nrows,
        title=("Corrimientos Doppler — V / Ni / Ti  ·  "
               "línea sólida: λ_obs (mínimo + parábola)  ·  "
               "discontinua: λ_lab"),
        showlegend=False,
    )
    return fig


# --- HTML rendering ----------------------------------------------------

CSS = """
:root {
  --primary: #870047;
  --secondary: #ff6aa7;
  --dark: #3a0020;
  --light: #ffc2da;
  --ink: #1a1014;
  --bg: #fff7fb;
  --card: #ffffff;
  --border: #ebd6df;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0;
  font-family: "DejaVu Sans", Helvetica, Arial, sans-serif;
  color: var(--ink); background: var(--bg);
}
header {
  padding: 36px 48px 28px;
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  color: white;
}
header h1 { margin: 0; font-size: 32px; letter-spacing: 0.5px; }
header p { margin: 6px 0 0; opacity: 0.92; font-size: 15px; }
main { padding: 28px 48px 80px; max-width: 1400px; margin: 0 auto; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px; margin: -50px 0 28px; }
.card { background: var(--card); border-radius: 14px;
  padding: 18px 20px; box-shadow: 0 6px 20px rgba(135,0,71,0.10);
  border: 1px solid var(--border); }
.card .label { color: var(--primary); font-weight: 600;
  font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; }
.card .value { font-size: 26px; font-weight: 700; margin-top: 4px; color: var(--ink); }
.card .sub   { font-size: 12px; color: var(--dark); opacity: 0.7; margin-top: 2px; }

h2 { margin: 38px 0 12px; font-size: 22px; color: var(--primary);
  border-bottom: 2px solid var(--border); padding-bottom: 8px; }
h3 { margin: 22px 0 8px; font-size: 16px; color: var(--dark); }
.subtitle { color: var(--ink); opacity: 0.7; margin: 0 0 14px; font-size: 14px; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 1000px) { .grid-2 { grid-template-columns: 1fr; } }

table.overview { width: 100%; border-collapse: collapse; background: white;
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 4px 14px rgba(135,0,71,0.06); border: 1px solid var(--border); }
table.overview th { background: var(--primary); color: white; padding: 10px;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; text-align: left; }
table.overview td { padding: 9px 10px; border-bottom: 1px solid var(--border);
  font-size: 13px; }
table.overview tr:last-child td { border-bottom: none; }
table.overview tr:hover td { background: #fff0f6; }

details.night { margin: 10px 0; background: white; border-radius: 12px;
  border: 1px solid var(--border); overflow: hidden; }
details.night > summary { padding: 14px 18px; cursor: pointer;
  font-weight: 600; color: var(--primary); list-style: none;
  display: flex; justify-content: space-between; align-items: center;
  background: linear-gradient(90deg, #fff0f6 0%, white 100%); }
details.night > summary::-webkit-details-marker { display: none; }
details.night > summary::after { content: "▼"; font-size: 10px;
  color: var(--primary); transition: transform 0.2s; }
details.night[open] > summary::after { transform: rotate(180deg); }
details.night .body { padding: 12px 18px 22px; }
.kpis { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 14px; }
.kpi { background: #fff0f6; padding: 8px 14px; border-radius: 8px;
  font-size: 13px; color: var(--dark); }
.kpi b { color: var(--primary); }

footer { text-align: center; color: var(--dark); opacity: 0.6;
  font-size: 12px; padding: 20px; }
"""


def _fmt(value, fmt: str, default: str = "—") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)


def overview_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No hay noches calibradas.</p>"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{r['date']}</td>"
            f"<td>{_fmt(r['degree'], 'd')}</td>"
            f"<td>{_fmt(r['rms_A'], '.4f')}</td>"
            f"<td>{_fmt(r['n_lines'], 'd')}</td>"
            f"<td>{_fmt(r['v_tel_kms'], '+.3f')}</td>"
            f"<td>{_fmt(r['v_rad_kms'], '+.3f')} ± {_fmt(r['v_rad_sem'], '.3f')}</td>"
            f"<td>{_fmt(r['R_median'], '.0f')}</td>"
            f"<td>{_fmt(r['snr'], '.0f')}</td>"
            f"<td>{r['seed'] or '—'}</td>"
            "</tr>"
        )
    head = ("<tr><th>Fecha</th><th>deg</th><th>RMS (Å)</th><th>N líneas</th>"
            "<th>v_tel (km/s)</th><th>v_rad (km/s)</th><th>R</th><th>SNR</th>"
            "<th>seed</th></tr>")
    return ("<table class='overview'><thead>" + head + "</thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def summary_cards(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    n = len(df)
    drange = f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}"
    rms = df["rms_A"].astype(float).mean()
    v = df["v_rad_kms"].dropna().astype(float)
    v_text = f"{v.mean():+.2f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.2f} km/s" \
        if len(v) else "—"
    R = df["R_median"].dropna().astype(float)
    R_text = f"{R.median():.0f}" if len(R) else "—"
    snr = df["snr"].dropna().astype(float)
    snr_text = f"{snr.median():.0f}" if len(snr) else "—"

    def card(label, value, sub=""):
        return (f"<div class='card'><div class='label'>{label}</div>"
                f"<div class='value'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>")

    return ("<div class='cards'>"
            + card("Noches calibradas", str(n), drange)
            + card("RMS medio", f"{rms:.4f} Å", "promedio entre noches")
            + card("v_rad estelar", v_text, "media ± dispersión inter-noche")
            + card("Resolución R", R_text, "mediana entre noches")
            + card("SNR", snr_text, "mediana entre noches")
            + "</div>")


def night_section(night: NightData, idx: int) -> str:
    """Detail accordion for one night."""
    sol, rv, sci = night.solution, night.rv, night.science
    res = (sci.get("resolution") or {})
    kpis = [
        f"<span class='kpi'>RMS <b>{_fmt(sol.get('rms'), '.4f')}</b> Å</span>",
        f"<span class='kpi'>N líneas <b>{_fmt(sol.get('n_lines'), 'd')}</b></span>",
        f"<span class='kpi'>v_rad <b>{_fmt(rv.get('v_mean_kms'), '+.3f')}</b> km/s"
        f" (n={rv.get('n_used','—')})</span>",
        f"<span class='kpi'>R <b>{_fmt(res.get('R_median'), '.0f')}</b></span>",
        f"<span class='kpi'>SNR <b>{_fmt(sci.get('snr'), '.0f')}</b></span>",
        f"<span class='kpi'>seed <b>{sol.get('seed_origin', '—')}</b></span>",
    ]

    figs_html = []
    if night.thar is not None:
        f = fig_spectrum(night.thar, f"Th-Ar  ·  {night.date}",
                         fill=True, lines_overlay=night.lines_used)
        figs_html.append(_fig_to_div(f, f"thar-{idx}"))
    if night.star is not None:
        spec = night.star
        lam_col = "lambda_rest" if "lambda_rest" in spec.columns else "lambda_cal"
        f = fig_spectrum(spec, f"Estrella  ·  {night.date}",
                         lambda_col=lam_col, fill=False)
        figs_html.append(_fig_to_div(f, f"star-{idx}"))
    rv_fig = fig_rv_fits(night)
    if rv_fig is not None:
        figs_html.append(_fig_to_div(rv_fig, f"rvfits-{idx}"))

    summary = (f"<span>{night.date}</span>"
               f"<span style='font-weight:400;font-size:13px;color:var(--dark);'>"
               f"v_rad = {_fmt(rv.get('v_mean_kms'), '+.3f')} km/s · "
               f"RMS = {_fmt(sol.get('rms'), '.4f')} Å · "
               f"R = {_fmt(res.get('R_median'), '.0f')}</span>")

    return (f"<details class='night'><summary>{summary}</summary>"
            f"<div class='body'><div class='kpis'>{''.join(kpis)}</div>"
            + "".join(figs_html) + "</div></details>")


def build_dashboard(results_dir: Path, out_path: Path) -> Path:
    nights = collect(results_dir)
    if not nights:
        raise RuntimeError(f"No calibrated nights found under {results_dir}.")

    df = overview_table(nights)

    parts: list[str] = []
    parts.append(_fig_to_div(fig_rms(df), "fig-rms"))
    parts.append(_fig_to_div(fig_n_lines(df), "fig-nlines"))
    if len(nights) >= 2:
        parts.append(_fig_to_div(fig_coefficients(nights), "fig-coef"))

    rv_parts = [_fig_to_div(fig_rv(df), "fig-rv"),
                _fig_to_div(fig_per_element(nights), "fig-rv-elem")]
    sci_parts = [_fig_to_div(fig_resolution(df), "fig-R"),
                 _fig_to_div(fig_snr(df), "fig-snr"),
                 _fig_to_div(fig_ew_heatmap(nights), "fig-ew")]

    nights_html = "".join(night_section(n, i) for i, n in enumerate(nights))

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>spectrogasm — dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>spectrogasm · dashboard</h1>
  <p>Calibración de longitud de onda + velocidad radial estelar + diagnóstico instrumental</p>
</header>
<main>
  {summary_cards(df)}

  <h2>1 · Resumen por noche</h2>
  <p class="subtitle">Una fila por noche calibrada. RMS = error residual del polinomio λ(pix); v_rad = velocidad estelar promediada sobre las 6 líneas V/Ni/Ti; R = poder resolutivo del espectrógrafo.</p>
  {overview_html(df)}

  <h2>2 · Calidad de la calibración</h2>
  <p class="subtitle">RMS, número de líneas usadas y deriva de los coeficientes del polinomio noche a noche. Saltos abruptos = eventos instrumentales; derivas suaves = drift óptico.</p>
  <div class="grid-2">{parts[0]}{parts[1]}</div>
  {parts[2] if len(parts) >= 3 else ""}

  <h2>3 · Velocidad radial estelar</h2>
  <p class="subtitle">v_rad media por noche (con SEM) y desglose por elemento. Los tres elementos deben coincidir dentro de las barras de error; sesgos sistemáticos delatan blends o errores residuales en λ.</p>
  <div class="grid-2">{rv_parts[0]}{rv_parts[1]}</div>

  <h2>4 · Diagnóstico y ciencia extra</h2>
  <p class="subtitle">Poder resolutivo R = λ/FWHM medido sobre el Th-Ar, SNR robusto del espectro estelar, y anchos equivalentes (mÅ) de las líneas estelares — proxy de abundancia química.</p>
  <div class="grid-2">{sci_parts[0]}{sci_parts[1]}</div>
  {sci_parts[2]}

  <h2>5 · Detalle por noche</h2>
  <p class="subtitle">Cada panel es interactivo: zoom, pan y hover. Los círculos rosa marcan las líneas atómicas usadas en la solución λ.</p>
  {nights_html}
</main>
<footer>Generado por <code>spectrogasm.dashboard</code> · paleta #870047 / #ff6aa7</footer>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv: list[str]) -> None:
    results_dir = Path(argv[0]) if argv else RESULTS_DIR
    out_path = Path(argv[1]) if len(argv) > 1 else results_dir / "dashboard.html"
    written = build_dashboard(results_dir, out_path)
    print(f"Dashboard escrito en: {written}")


if __name__ == "__main__":
    main(sys.argv[1:])
