import os
import re
import csv
from collections import defaultdict

PROFILING_DIR = os.path.join(
    os.path.dirname(__file__),
    "symphony_of_tears_features", "results_profiling"
)
OUTPUT_SUMMARY = "runtime_summary_new.csv"
OUTPUT_FLEXDTW = "runtime_chunk_flexdtw_new.csv"


def parse_folder_name(name):
    m = re.match(r'^(dense|sparse|flexdtw)_P(\d+)_L(\d+)_trial(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def collect_data(profiling_dir):
    # keyed by (mode, P, L) -> list of trial dicts
    trials = defaultdict(list)

    for folder in sorted(os.listdir(profiling_dir)):
        parsed = parse_folder_name(folder)
        if parsed is None:
            continue
        mode, P, L, trial = parsed
        folder_path = os.path.join(profiling_dir, folder)

        trial_data = {
            "trial": trial,
            "chunk_flexdtw": read_csv_rows(os.path.join(folder_path, "chunk_flexdtw.csv")),
            "initialize_chunks": read_csv_rows(os.path.join(folder_path, "initialize_chunks.csv")),
            "dp_fill_chunks": read_csv_rows(os.path.join(folder_path, "dp_fill_chunks.csv")),
            "stage_2_backtrace": read_csv_rows(os.path.join(folder_path, "stage_2_backtrace_compatible.csv")),
            "flexdtw_runtime": read_csv_rows(os.path.join(folder_path, "flexdtw_runtime.csv")),
        }
        trials[(mode, P, L)].append(trial_data)

    return trials


def avg(values):
    return sum(values) / len(values) if values else None


def compute_summary(trials):
    summary_rows = []
    flexdtw_rows = []

    for (mode, P, L), trial_list in sorted(trials.items()):

        # --- FlexDTW full-matrix runtime (only present for mode == "flexdtw") ---
        flexdtw_totals = []
        for t in trial_list:
            rows_ft = t["flexdtw_runtime"]
            if rows_ft:
                flexdtw_totals.append(float(rows_ft[0]["elapsed_seconds"]))
        avg_flexdtw = avg(flexdtw_totals)

        # --- Metric 2: initialize_chunks total elapsed per trial, then average ---
        init_totals = []
        for t in trial_list:
            total = sum(float(r["elapsed_seconds"]) for r in t["initialize_chunks"])
            init_totals.append(total)
        avg_init = avg(init_totals) if init_totals else None

        # --- Metric 3: dp_fill_chunks total elapsed per trial, then average ---
        dp_totals = []
        for t in trial_list:
            total = sum(float(r["elapsed_seconds"]) for r in t["dp_fill_chunks"])
            dp_totals.append(total)
        avg_dp = avg(dp_totals) if dp_totals else None

        # --- Metric 4: stage_2 top_scan + right_scan + backtrace_stitch total, then average ---
        stage2_totals = []
        for t in trial_list:
            rows = {r["phase"]: float(r["elapsed_seconds"]) for r in t["stage_2_backtrace"]}
            if rows:
                total = sum(rows.get(p, 0.0) for p in ("top_scan", "right_scan", "backtrace_stitch"))
                stage2_totals.append(total)
        avg_stage2 = avg(stage2_totals) if stage2_totals else None

        # Count unique (chunk_i, chunk_j) pairs — use the first trial that has data.
        num_chunks = 0
        for t in trial_list:
            rows_cf = t["chunk_flexdtw"]
            if rows_cf:
                num_chunks = len({(r["chunk_i"], r["chunk_j"]) for r in rows_cf})
                break

        summary_rows.append({
            "mode": mode,
            "P": P,
            "L": L,
            "num_trials": len(trial_list),
            "num_chunks": num_chunks,
            "avg_flexdtw_total_s": round(avg_flexdtw, 6) if avg_flexdtw is not None else "",
            "avg_initialize_chunks_total_s": round(avg_init, 6) if avg_init is not None else "",
            "avg_dp_fill_chunks_total_s": round(avg_dp, 6) if avg_dp is not None else "",
            "avg_stage2_backtrace_total_s": round(avg_stage2, 6) if avg_stage2 is not None else "",
        })

        # --- Metric 1: per-chunk flexdtw average across trials ---
        chunk_times = defaultdict(list)
        for t in trial_list:
            for r in t["chunk_flexdtw"]:
                key = (int(r["chunk_i"]), int(r["chunk_j"]))
                chunk_times[key].append(float(r["elapsed_seconds"]))

        for (ci, cj), times in sorted(chunk_times.items()):
            flexdtw_rows.append({
                "mode": mode,
                "P": P,
                "L": L,
                "chunk_i": ci,
                "chunk_j": cj,
                "num_trials": len(times),
                "avg_elapsed_seconds": round(avg(times), 9),
            })

    return summary_rows, flexdtw_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}  ({len(rows)} rows)")


def pivot_summary(summary_rows):
    """Pivot so dense, sparse, and flexdtw are side-by-side columns for each (P, L)."""
    parflex_metrics = [
        "avg_initialize_chunks_total_s",
        "avg_dp_fill_chunks_total_s",
        "avg_stage2_backtrace_total_s",
    ]
    by_pl = {}
    for r in summary_rows:
        key = (r["P"], r["L"])
        by_pl.setdefault(key, {})[r["mode"]] = r

    rows = []
    for (P, L) in sorted(by_pl.keys()):
        d  = by_pl[(P, L)].get("dense", {})
        s  = by_pl[(P, L)].get("sparse", {})
        fx = by_pl[(P, L)].get("flexdtw", {})
        # num_chunks is mode-independent; prefer dense, fall back to sparse.
        num_chunks = d.get("num_chunks", s.get("num_chunks", ""))
        row = {
            "P": P,
            "L": L,
            "num_chunks": num_chunks,
            "flexdtw_avg_total_s": fx.get("avg_flexdtw_total_s", ""),
        }
        for m in parflex_metrics:
            row[f"dense_{m}"]  = d.get(m, "")
            row[f"sparse_{m}"] = s.get(m, "")
        rows.append(row)

    fieldnames = (
        ["P", "L", "num_chunks", "flexdtw_avg_total_s"]
        + [f"{mode}_{m}" for m in parflex_metrics for mode in ("dense", "sparse")]
    )
    return rows, fieldnames


def pivot_flexdtw(flexdtw_rows):
    """Pivot so dense and sparse avg_elapsed_seconds are side-by-side for each (P, L, chunk_i, chunk_j)."""
    by_key = {}
    for r in flexdtw_rows:
        if r["mode"] == "flexdtw":
            continue  # flexdtw has no per-chunk data
        key = (r["P"], r["L"], r["chunk_i"], r["chunk_j"])
        by_key.setdefault(key, {})[r["mode"]] = r["avg_elapsed_seconds"]

    rows = []
    for (P, L, ci, cj) in sorted(by_key.keys()):
        entry = by_key[(P, L, ci, cj)]
        rows.append({
            "P": P,
            "L": L,
            "chunk_i": ci,
            "chunk_j": cj,
            "dense_avg_elapsed_seconds":  entry.get("dense", ""),
            "sparse_avg_elapsed_seconds": entry.get("sparse", ""),
        })

    fieldnames = ["P", "L", "chunk_i", "chunk_j", "dense_avg_elapsed_seconds", "sparse_avg_elapsed_seconds"]
    return rows, fieldnames


def main():
    trials = collect_data(PROFILING_DIR)
    print(f"Found {len(trials)} (mode, P, L) combinations across all trials.")

    summary_rows, flexdtw_rows = compute_summary(trials)

    pivoted_summary, summary_fields = pivot_summary(summary_rows)
    write_csv(OUTPUT_SUMMARY, pivoted_summary, fieldnames=summary_fields)

    pivoted_flexdtw, flexdtw_fields = pivot_flexdtw(flexdtw_rows)
    write_csv(OUTPUT_FLEXDTW, pivoted_flexdtw, fieldnames=flexdtw_fields)


if __name__ == "__main__":
    main()
