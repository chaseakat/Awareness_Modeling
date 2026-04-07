import csv

REGIME_FILE = "regime_map.csv"

states = {
    "awake_rest": {"alpha": (8.0, 12.0), "chi": (0.40, 0.60)},
    "n1_n2": {"alpha": (6.0, 9.0), "chi": (0.35, 0.60)},
    "n3_sws": {"alpha": (0.5, 4.0), "chi": (0.20, 0.70)},
    "rem": {"alpha": (6.0, 10.0), "chi": (0.40, 0.65)},
}

with open(REGIME_FILE) as f:
    rows = list(csv.DictReader(f))


def filter_bins(alpha_range, chi_range):
    selected = []
    for r in rows:
        a = float(r["alpha_center_hz"])
        c = float(r["chi_center"])
        if alpha_range[0] <= a <= alpha_range[1] and chi_range[0] <= c <= chi_range[1]:
            selected.append(r)
    return selected


print("=== SLEEP STATE PROJECTION ===\n")
for state, cfg in states.items():
    bins = filter_bins(cfg["alpha"], cfg["chi"])
    if not bins:
        print(f"{state}: NO MATCHING BINS\n")
        continue
    scores = [float(r["mean_score"]) for r in bins]
    loops = [r["dominant_loop"] for r in bins]
    avg_score = sum(scores) / len(scores)
    counts = {}
    for l in loops:
        counts[l] = counts.get(l, 0) + 1
    dominant_loop = max(counts, key=counts.get)
    print(f"{state.upper()}:")
    print(f"  bins: {len(bins)}")
    print(f"  avg_score: {round(avg_score, 4)}")
    print(f"  dominant_loop: {dominant_loop}")
    print(f"  loop_distribution: {counts}")
    print()
