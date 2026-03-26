import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("runtime_summary_new.csv")

# Total pipeline time = initialize + dp_fill + stage2_backtrace
for mode in ("dense", "sparse"):
    df[f"{mode}_total_s"] = (
        df[f"{mode}_avg_initialize_chunks_total_s"]
        + df[f"{mode}_avg_dp_fill_chunks_total_s"]
        + df[f"{mode}_avg_stage2_backtrace_total_s"]
    )

# Percentage speedup of sparse over dense: (dense - sparse) / dense * 100
df["pct_change"] = (df["dense_total_s"] - df["sparse_total_s"]) / df["dense_total_s"] * 100

# Only rows with more than 1 chunk
df = df[df["num_chunks"] > 1].copy()

L_VALUES = sorted(df["L"].unique())
COLORS   = {2000: "#1f5fa6", 4000: "#1a7a3c", 6000: "#b51f1f"}

# Build one group per unique (P, num_chunks) — x-axis label shows both
df = df.sort_values(["P", "L"])
groups   = df[["P", "L", "num_chunks"]].drop_duplicates().sort_values(["P", "L"])
x_labels = [f"P{row.P}\nL{row.L}\n({int(row.num_chunks)} chunks)"
            for row in groups.itertuples()]
x        = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(max(12, len(x_labels) * 0.9), 6))
fig.suptitle(
    "Sparse vs Dense Speedup: (Dense − Sparse) / Dense  [% faster]\n"
    "Only configurations with >1 chunk",
    fontsize=13, fontweight="bold",
)

bar_width = 0.7
seen_labels = set()
for i, row in enumerate(groups.itertuples()):
    val = df.loc[(df["P"] == row.P) & (df["L"] == row.L), "pct_change"].values[0]
    color = COLORS[row.L]
    label = f"L{row.L}" if row.L not in seen_labels else "_nolegend_"
    seen_labels.add(row.L)
    ax.bar(i, val, width=bar_width, color=color, edgecolor="white", linewidth=0.5,
           label=label)

# Horizontal reference line at 0
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

# Value labels on each bar
for i, row in enumerate(groups.itertuples()):
    val = df.loc[(df["P"] == row.P) & (df["L"] == row.L), "pct_change"].values[0]
    va  = "bottom" if val >= 0 else "top"
    offset = 0.5 if val >= 0 else -0.5
    ax.text(i, val + offset, f"{val:.1f}%", ha="center", va=va, fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_xlabel("Configuration  (P / L / num_chunks)", fontsize=11)
ax.set_ylabel("% faster  (positive = sparse is faster)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), title="Chunk Length (L)",
          fontsize=9, title_fontsize=9)

plt.tight_layout()
out = "runtime_summary_barchart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
