#!/usr/bin/env python3
"""Bài 4: Manual single-linkage hierarchical clustering (8 students).

Simulates 8 students (SV1..SV8) with scores in two subjects.  Computes
Euclidean distance matrix, then prints every merge step manually
(without SciPy shortcuts for the merge logic).  Cross-checks against
scipy.cluster.hierarchy.linkage(method="single").
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

from common import OUTPUT_DIR, ensure_dirs, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ---------------------------------------------------------------------------
# Student data
# ---------------------------------------------------------------------------

STUDENTS = ["SV1", "SV2", "SV3", "SV4", "SV5", "SV6", "SV7", "SV8"]
# (Math, Literature) scores
SCORES = np.array([
    [2, 8],
    [3, 9],
    [3, 7],
    [5, 5],
    [7, 3],
    [8, 2],
    [8, 4],
    [9, 1],
], dtype=float)


# ---------------------------------------------------------------------------
# Manual single linkage
# ---------------------------------------------------------------------------

def format_cluster(c: frozenset) -> str:
    return "{" + ", ".join(sorted(c)) + "}"


def build_pair_distance(points: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for a, b in combinations(points, 2):
        d = float(np.linalg.norm(points[a] - points[b]))
        lookup[tuple(sorted((a, b)))] = d
    return lookup


def min_cross_distance(
    cA: frozenset,
    cB: frozenset,
    pdict: dict[tuple[str, str], float],
) -> tuple[float, tuple[str, str]]:
    """Return (min distance, closest pair) between two clusters (single linkage)."""
    best_d = math.inf
    best_pair: tuple[str, str] | None = None
    for x, y in combinations(sorted(cA | cB), 2):
        if (x in cA and y in cB) or (x in cB and y in cA):
            d = pdict[(x, y) if x < y else (y, x)]
            canonical = (x, y) if x < y else (y, x)
            if d < best_d or (d == best_d and (best_pair is None or canonical < best_pair)):
                best_d = d
                best_pair = canonical
    assert best_pair is not None
    return best_d, best_pair


def current_dist_matrix(
    clusters: list[frozenset],
    pdict: dict[tuple[str, str], float],
) -> pd.DataFrame:
    names = [format_cluster(c) for c in clusters]
    n = len(clusters)
    data = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        d, _ = min_cross_distance(clusters[i], clusters[j], pdict)
        data[i, j] = data[j, i] = d
    return pd.DataFrame(data, index=names, columns=names)


def run_single_linkage(
    labels: list[str],
    pdict: dict[tuple[str, str], float],
) -> list[dict]:
    print("\n=== Bài 4: Manual single linkage — 8 students ===")
    clusters = [frozenset({lbl}) for lbl in labels]
    print("\nInitial distance matrix:")
    print(current_dist_matrix(clusters, pdict).to_string())

    merges = []
    step = 0
    while len(clusters) > 1:
        step += 1
        n = len(clusters)
        best: tuple[float, int, int, tuple[str, str]] | None = None
        for i, j in combinations(range(n), 2):
            d, pair = min_cross_distance(clusters[i], clusters[j], pdict)
            key_a, key_b = format_cluster(clusters[i]), format_cluster(clusters[j])
            tie = (key_a, key_b)
            cand = (d, tie, i, j, pair)
            if best is None or cand[:2] < best[:2]:
                best = cand
        assert best is not None
        d_val, _, i, j, pair = best
        a, b = clusters[i], clusters[j]
        merged = a | b

        merges.append({
            "step": step,
            "cluster_a": sorted(a),
            "cluster_b": sorted(b),
            "distance": round(d_val, 4),
            "closest_pair": list(pair),
            "members": sorted(merged),
        })

        print(
            f"\nStep {step}: merge {format_cluster(a)} + {format_cluster(b)} "
            f"at d={d_val:.4f}  (closest pair: {pair})"
        )

        new_clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        new_clusters.append(merged)
        new_clusters.sort(key=lambda c: sorted(c))
        clusters = new_clusters

        print(f"  Updated matrix after step {step}:")
        print(current_dist_matrix(clusters, pdict).to_string())

    print("\nMerge order summary:")
    for m in merges:
        print(f"  Step {m['step']}: {m['cluster_a']} + {m['cluster_b']} -> {m['members']} (d={m['distance']})")

    return merges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    points = {s: SCORES[i] for i, s in enumerate(STUDENTS)}
    pdict = build_pair_distance(points)

    # Manual runs
    merges = run_single_linkage(STUDENTS, pdict)

    # Save merge history
    merge_df = pd.DataFrame(merges)
    merge_df.to_csv(RESULTS_DIR / "bai_04_merge_history.csv", index=False)
    print(f"\nSaved merge history to {RESULTS_DIR / 'bai_04_merge_history.csv'}")

    # Cross-check with SciPy
    condensed = [pdict[tuple(sorted(p))] for p in combinations(STUDENTS, 2)]
    Z = linkage(condensed, method="single")
    scipy_dists = [round(float(z[2]), 4) for z in Z]
    manual_dists = [m["distance"] for m in merges]
    print(f"\nCross-check against scipy:")
    print(f"  Manual distances:   {manual_dists}")
    print(f"  SciPy distances:    {scipy_dists}")
    print(f"  Match: {manual_dists == scipy_dists}")

    # Score scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(SCORES[:, 0], SCORES[:, 1], c="royalblue", s=100, edgecolors="k")
    for i, s in enumerate(STUDENTS):
        ax.text(SCORES[i, 0] + 0.15, SCORES[i, 1] + 0.15, s, fontsize=10)
    ax.set_xlabel("Math")
    ax.set_ylabel("Literature")
    ax.set_title("Bài 4 — 8 student scores")
    savefig("bai_04_student_scores.png")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(Z, labels=STUDENTS, leaf_rotation=0, leaf_font_size=10,
               color_threshold=0, above_threshold_color="#444444", ax=ax)
    ax.set_title("Bài 4 — Single linkage dendrogram (8 students)")
    ax.set_ylabel("Distance")
    savefig("bai_04_dendrogram.png")

    # Metrics
    from common import save_metrics
    save_metrics([{
        "dataset": "8_students_math_literature",
        "n_points": len(STUDENTS),
        "manual_merge_dists": str(manual_dists),
        "scipy_match": str(manual_dists == scipy_dists),
    }], "bai_04_metrics.csv")


if __name__ == "__main__":
    main()
