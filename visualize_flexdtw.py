import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("runtime_chunk_flexdtw.csv")

# Average all per-chunk times for each (P, L) → one number per (mode, P, L)
summary = (
    df.groupby(["P", "L"])[["dense_avg_elapsed_seconds", "sparse_avg_elapsed_seconds"]]
    .mean()
    .reset_index()
)

L_VALUES = sorted(summary["L"].unique())
P_VALUES = sorted(summary["P"].unique())

# 6 bars per P group: L2000 dense, L2000 sparse, L4000 dense, L4000 sparse, L6000 dense, L6000 sparse
BARS = [
    (2000, "dense",  "#1f5fa6", "L2000 Dense"),
    (2000, "sparse", "#7ab3e0", "L2000 Sparse"),
    (4000, "dense",  "#1a7a3c", "L4000 Dense"),
    (4000, "sparse", "#7ecf99", "L4000 Sparse"),
    (6000, "dense",  "#b51f1f", "L6000 Dense"),
    (6000, "sparse", "#f08080", "L6000 Sparse"),
]

n_bars = len(BARS)
bar_width = 0.13
x = np.arange(len(P_VALUES))
offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width

fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle(
    "Average Chunk FlexDTW Run Time vs P\n(mean over all chunks and 10 trials)",
    fontsize=14, fontweight="bold"
)

for offset, (L, mode, color, label) in zip(offsets, BARS):
    col = f"{mode}_avg_elapsed_seconds"
    vals = [
        summary.loc[(summary["P"] == P) & (summary["L"] == L), col].values[0]
        for P in P_VALUES
    ]
    ax.bar(x + offset, vals, width=bar_width, color=color, label=label,
           edgecolor="white", linewidth=0.4)

# Shade alternating P groups
for i in range(0, len(P_VALUES), 2):
    ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.06, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels([f"P{p}" for p in P_VALUES], fontsize=10)
ax.set_xlabel("P (number of points)", fontsize=12)
ax.set_ylabel("Avg Chunk Time (s)", fontsize=12)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}s"))
ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(ncol=3, fontsize=9, loc="upper left", framealpha=0.8,
          columnspacing=0.8, handlelength=1.2)

plt.tight_layout()
out = "flexdtw_avg_chunk_time_vs_P.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
