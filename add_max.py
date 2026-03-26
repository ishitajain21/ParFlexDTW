import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

CSV_PATH = "runtime_analysis/Results - MIR Benchmarks - Runtime analysis .csv"
OUT_PATH = "runtime_analysis/runtime_visuals.html"

df = pd.read_csv(CSV_PATH, skiprows=1)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["P"])
df["P"] = df["P"].astype(int)
df["L"] = df["L"].astype(int)

L_VALUES = sorted(df["L"].unique())
L_COLORS = {2000: "#1f77b4", 4000: "#ff7f0e", 6000: "#2ca02c"}
METHOD_COLORS = {"FlexDTW": "#e74c3c", "Dense ParFlex": "#3498db", "Sparse ParFlex": "#2ecc71"}

figs = []

# ── 1. Total Runtime vs P (log-log, one line per L × method) ─────────────────
fig1 = go.Figure()
for L in L_VALUES:
    sub = df[df["L"] == L].sort_values("P")
    dash_styles = {"FlexDTW": "dot", "Dense ParFlex": "dash", "Sparse ParFlex": "solid"}
    for method, col in [("FlexDTW", "flexdtw_avg_total_s"),
                         ("Dense ParFlex", "total runtime dense"),
                         ("Sparse ParFlex", "total runtime sparse")]:
        fig1.add_trace(go.Scatter(
            x=sub["P"], y=sub[col],
            mode="lines+markers",
            name=f"{method} (L={L})",
            line=dict(color=METHOD_COLORS[method], dash=dash_styles[method], width=2),
            marker=dict(size=7),
            legendgroup=method,
        ))

fig1.update_layout(
    title="Total Runtime vs Signal Length P (log–log)",
    xaxis=dict(title="P (signal length)", type="log"),
    yaxis=dict(title="Runtime (s)", type="log"),
    legend=dict(groupclick="toggleitem"),
    hovermode="x unified",
    template="plotly_white",
)
figs.append(("Total Runtime vs P", fig1))

# ── 2. Speedup of ParFlex over FlexDTW ───────────────────────────────────────
fig2 = go.Figure()
for L in L_VALUES:
    sub = df[df["L"] == L].sort_values("P")
    for method, col in [("Dense ParFlex", "total runtime dense"),
                         ("Sparse ParFlex", "total runtime sparse")]:
        speedup = sub["flexdtw_avg_total_s"] / sub[col]
        fig2.add_trace(go.Scatter(
            x=sub["P"], y=speedup,
            mode="lines+markers",
            name=f"{method} (L={L})",
            line=dict(color=METHOD_COLORS[method],
                      dash="dash" if method == "Dense ParFlex" else "solid", width=2),
            marker=dict(size=7),
            legendgroup=method,
        ))
fig2.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="1× (no speedup)")
fig2.update_layout(
    title="Speedup over FlexDTW (FlexDTW runtime / ParFlex runtime)",
    xaxis=dict(title="P (signal length)", type="log"),
    yaxis=dict(title="Speedup factor (×)", type="log"),
    legend=dict(groupclick="toggleitem"),
    hovermode="x unified",
    template="plotly_white",
)
figs.append(("Speedup over FlexDTW", fig2))

# ── 3. Number of Chunks vs P ──────────────────────────────────────────────────
fig3 = go.Figure()
for L in L_VALUES:
    sub = df[df["L"] == L].sort_values("P")
    fig3.add_trace(go.Scatter(
        x=sub["P"], y=sub["num_chunks"],
        mode="lines+markers",
        name=f"L={L}",
        line=dict(color=L_COLORS[L], width=2),
        marker=dict(size=8),
    ))
fig3.update_layout(
    title="Number of Chunks vs P",
    xaxis=dict(title="P", type="log"),
    yaxis=dict(title="num_chunks", type="log"),
    template="plotly_white",
    hovermode="x unified",
)
figs.append(("Number of Chunks", fig3))

# ── 4. Max Per-Chunk Runtime: Dense vs Sparse ─────────────────────────────────
fig4 = make_subplots(rows=1, cols=len(L_VALUES),
                     subplot_titles=[f"L = {L}" for L in L_VALUES],
                     shared_yaxes=True)
for idx, L in enumerate(L_VALUES, start=1):
    sub = df[df["L"] == L].sort_values("P")
    fig4.add_trace(go.Scatter(
        x=sub["P"], y=sub["max runtime for 1 chunk dense"],
        mode="lines+markers", name="Dense", line=dict(color=METHOD_COLORS["Dense ParFlex"], width=2),
        showlegend=(idx == 1), legendgroup="dense",
    ), row=1, col=idx)
    fig4.add_trace(go.Scatter(
        x=sub["P"], y=sub["max runtime for 1 chunk sparse"],
        mode="lines+markers", name="Sparse", line=dict(color=METHOD_COLORS["Sparse ParFlex"], width=2),
        showlegend=(idx == 1), legendgroup="sparse",
    ), row=1, col=idx)
    fig4.update_xaxes(title_text="P", type="log", row=1, col=idx)
fig4.update_yaxes(title_text="Max chunk runtime (s)", type="log", row=1, col=1)
fig4.update_layout(
    title="Max Runtime for a Single Chunk: Dense vs Sparse",
    template="plotly_white",
    hovermode="x unified",
)
figs.append(("Max Chunk Runtime", fig4))

# ── 5. Stage breakdown stacked bar (Dense) ───────────────────────────────────
def stage_breakdown_fig(method_label, col_init, col_dp, col_bt, col_total):
    sub_all = []
    for L in L_VALUES:
        sub = df[df["L"] == L].sort_values("P").copy()
        sub["label"] = sub["P"].astype(str) + f" (L={L})"
        sub_all.append(sub)
    sub_all = pd.concat(sub_all)

    fig = go.Figure()
    stage_cols = {
        "Init": col_init,
        "DP Fill": col_dp,
        "Backtrace": col_bt,
    }
    palette = ["#5b9bd5", "#ed7d31", "#70ad47"]
    for color, (stage, col) in zip(palette, stage_cols.items()):
        fig.add_trace(go.Bar(
            x=sub_all["label"], y=sub_all[col],
            name=stage, marker_color=color,
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Runtime Stage Breakdown – {method_label}",
        xaxis=dict(title="P (L)", tickangle=45),
        yaxis=dict(title="Time (s)"),
        template="plotly_white",
        legend=dict(title="Stage"),
    )
    return fig

fig5 = stage_breakdown_fig(
    "Dense ParFlex",
    "dense_avg_initialize_chunks_total_s",
    "dense_avg_dp_fill_chunks_total_s",
    "dense_avg_stage2_backtrace_total_s",
    "total runtime dense",
)
figs.append(("Stage Breakdown Dense", fig5))

fig6 = stage_breakdown_fig(
    "Sparse ParFlex",
    "sparse_avg_initialize_chunks_total_s",
    "sparse_avg_dp_fill_chunks_total_s",
    "sparse_avg_stage2_backtrace_total_s",
    "total runtime sparse",
)
figs.append(("Stage Breakdown Sparse", fig6))

# ── 6. % of Total Runtime that is Stage 1 ────────────────────────────────────
fig7 = go.Figure()
for L in L_VALUES:
    sub = df[df["L"] == L].sort_values("P")
    fig7.add_trace(go.Scatter(
        x=sub["P"], y=sub["percent of total run time which is stage 1 dense"],
        mode="lines+markers",
        name=f"Dense (L={L})",
        line=dict(color=METHOD_COLORS["Dense ParFlex"],
                  dash="dash" if L == 4000 else ("dot" if L == 6000 else "solid"), width=2),
        marker=dict(size=7),
    ))
    fig7.add_trace(go.Scatter(
        x=sub["P"], y=sub["percent of total run time which is stage 1 sparse"],
        mode="lines+markers",
        name=f"Sparse (L={L})",
        line=dict(color=METHOD_COLORS["Sparse ParFlex"],
                  dash="dash" if L == 4000 else ("dot" if L == 6000 else "solid"), width=2),
        marker=dict(size=7),
    ))
fig7.update_layout(
    title="% of Total Runtime Spent in Stage 1 (Chunking / Init)",
    xaxis=dict(title="P", type="log"),
    yaxis=dict(title="Stage 1 %", range=[0, 105]),
    template="plotly_white",
    hovermode="x unified",
)
figs.append(("Stage 1 % of Total", fig7))

# ── Assemble into one HTML file ───────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

html_parts = ["""
<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>ParFlex Runtime Analysis</title>
<style>
  body { font-family: sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }
  h1   { text-align: center; color: #2c3e50; }
  .card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1);
          margin: 24px auto; padding: 16px; max-width: 1100px; }
  .nav { text-align: center; margin-bottom: 20px; }
  .nav a { margin: 0 10px; color: #3498db; text-decoration: none; font-weight: bold; }
</style></head><body>
<h1>ParFlex Runtime Analysis</h1>
<div class="nav">
"""]

for title, _ in figs:
    anchor = title.replace(" ", "_")
    html_parts.append(f'  <a href="#{anchor}">{title}</a>')

html_parts.append("</div>")

for title, fig in figs:
    anchor = title.replace(" ", "_")
    html_parts.append(f'<div class="card" id="{anchor}">')
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if title == figs[0][0] else False))
    html_parts.append("</div>")

html_parts.append("</body></html>")

with open(OUT_PATH, "w") as f:
    f.write("\n".join(html_parts))

print(f"Saved: {OUT_PATH}")
