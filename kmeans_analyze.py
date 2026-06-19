#!/usr/bin/env python3
"""
Cluster HiBench workloads by their resource-usage profile.

Reads logs/run_all/resource_usage.csv (produced by monitor_resources.sh while
run_all_flink_batch.sh was running), aggregates per-workload stats (avg/max
CPU, memory, network), then runs K-Means (pure stdlib, no numpy/sklearn
required) to group workloads into clusters such as "CPU-heavy",
"network-heavy", "idle/light", etc.

Usage:
    python3 kmeans_analyze.py [--csv path] [--k 3] [--iters 100]
"""

import argparse
import csv
import random
from collections import defaultdict

FEATURES = ["avg_cpu", "max_cpu", "avg_mem", "max_mem", "avg_net"]


def load_and_aggregate(csv_path):
    per_workload = defaultdict(lambda: {"cpu": [], "mem": [], "net": []})
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl = row["workload"]
            if wl in ("idle", "unknown", "done"):
                continue
            try:
                cpu = float(row["cpu_pct"])
                mem = float(row["mem_pct"])
                net = float(row["net_rx_kbps"]) + float(row["net_tx_kbps"])
            except ValueError:
                continue
            per_workload[wl]["cpu"].append(cpu)
            per_workload[wl]["mem"].append(mem)
            per_workload[wl]["net"].append(net)

    rows = []
    for wl, d in per_workload.items():
        if not d["cpu"]:
            continue
        rows.append({
            "workload": wl,
            "avg_cpu": sum(d["cpu"]) / len(d["cpu"]),
            "max_cpu": max(d["cpu"]),
            "avg_mem": sum(d["mem"]) / len(d["mem"]),
            "max_mem": max(d["mem"]),
            "avg_net": sum(d["net"]) / len(d["net"]),
            "samples": len(d["cpu"]),
        })
    return rows


def zscore_normalize(rows):
    stats = {}
    for feat in FEATURES:
        vals = [r[feat] for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5 or 1.0
        stats[feat] = (mean, std)

    vectors = []
    for r in rows:
        vec = [(r[feat] - stats[feat][0]) / stats[feat][1] for feat in FEATURES]
        vectors.append(vec)
    return vectors


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(vectors, k, iters=100, seed=42):
    rng = random.Random(seed)
    centroids = rng.sample(vectors, k)
    assignments = [0] * len(vectors)

    for _ in range(iters):
        changed = False
        for i, v in enumerate(vectors):
            best, best_d = 0, float("inf")
            for c_idx, c in enumerate(centroids):
                d = dist2(v, c)
                if d < best_d:
                    best, best_d = c_idx, d
            if assignments[i] != best:
                assignments[i] = best
                changed = True

        new_centroids = []
        for c_idx in range(k):
            members = [v for i, v in enumerate(vectors) if assignments[i] == c_idx]
            if not members:
                new_centroids.append(centroids[c_idx])
                continue
            dim = len(members[0])
            new_centroids.append([sum(m[d] for m in members) / len(members) for d in range(dim)])
        centroids = new_centroids

        if not changed:
            break

    return assignments, centroids


CLUSTER_COLORS = ["\033[36m", "\033[33m", "\033[32m", "\033[35m", "\033[31m", "\033[34m"]
RESET = "\033[0m"


def save_matplotlib_chart(rows, assignments, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[chart] matplotlib not installed — run: pip3 install matplotlib")
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for c in sorted(set(assignments)):
        xs = [r["avg_cpu"] for r, a in zip(rows, assignments) if a == c]
        ys = [r["avg_net"] for r, a in zip(rows, assignments) if a == c]
        labels = [r["workload"] for r, a in zip(rows, assignments) if a == c]
        ax.scatter(xs, ys, color=cmap(c % 10), label=f"cluster {c}", s=80)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("avg CPU %")
    ax.set_ylabel("avg network (KB/s)")
    ax.set_title("HiBench workloads clustered by resource usage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[chart] saved to {out_path}")
    return True


def bar(value, max_value, width=40):
    n = int(round((value / max_value) * width)) if max_value > 0 else 0
    return "█" * n


def print_ascii_chart(rows_by_cluster, metric, title, no_color=False):
    print(f"\n{title}")
    print("-" * 80)
    all_vals = [r[metric] for members in rows_by_cluster.values() for r in members]
    max_value = max(all_vals) if all_vals else 1
    for c in sorted(rows_by_cluster):
        color = "" if no_color else CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        reset = "" if no_color else RESET
        for r in sorted(rows_by_cluster[c], key=lambda x: -x[metric]):
            label = f"[{c}] {r['workload']}"
            b = bar(r[metric], max_value)
            print(f"{label:<32} {color}{b}{reset} {r[metric]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/run_all/resource_usage.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--chart", action="store_true", help="print ASCII bar charts after the table")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors in chart output")
    ap.add_argument("--plot", metavar="PNG_PATH", help="save a matplotlib scatter plot (cpu vs net) to this path")
    args = ap.parse_args()

    rows = load_and_aggregate(args.csv)
    if len(rows) < args.k:
        print(f"Only {len(rows)} workloads found — reduce --k to at most that many.")
        return

    vectors = zscore_normalize(rows)
    assignments, _ = kmeans(vectors, args.k, args.iters)

    clusters = defaultdict(list)
    for r, c in zip(rows, assignments):
        clusters[c].append(r)

    print(f"{'workload':<28} {'cluster':<8} {'avg_cpu%':<10} {'max_cpu%':<10} {'avg_mem%':<10} {'avg_net(KB/s)':<14}")
    print("-" * 80)
    for c in sorted(clusters):
        for r in sorted(clusters[c], key=lambda x: -x["avg_cpu"]):
            print(f"{r['workload']:<28} {c:<8} {r['avg_cpu']:<10.2f} {r['max_cpu']:<10.2f} "
                  f"{r['avg_mem']:<10.2f} {r['avg_net']:<14.2f}")

    print("\nCluster summary:")
    for c in sorted(clusters):
        members = clusters[c]
        avg_cpu = sum(m["avg_cpu"] for m in members) / len(members)
        avg_mem = sum(m["avg_mem"] for m in members) / len(members)
        avg_net = sum(m["avg_net"] for m in members) / len(members)
        print(f"  cluster {c}: {len(members)} workloads — "
              f"avg_cpu={avg_cpu:.1f}%  avg_mem={avg_mem:.1f}%  avg_net={avg_net:.1f}KB/s")

    if args.chart:
        print_ascii_chart(clusters, "avg_cpu", "CPU usage by workload (grouped by cluster)", args.no_color)
        print_ascii_chart(clusters, "avg_net", "Network usage (KB/s) by workload (grouped by cluster)", args.no_color)

    if args.plot:
        save_matplotlib_chart(rows, assignments, args.plot)


if __name__ == "__main__":
    main()
