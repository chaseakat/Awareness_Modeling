import csv
import statistics
from collections import defaultdict

INPUT = "diversified_train.csv"
OUT_CSV = "regime_map.csv"
OUT_TXT = "regime_map.txt"

CHI_MIN, CHI_MAX, CHI_STEP = 0.20, 0.80, 0.02
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = 0.5, 14.0, 0.25

def bin_index(x, lo, hi, step):
    if x < lo or x >= hi:
        return None
    return int((x - lo) // step)

def bin_center(idx, lo, step):
    return lo + (idx + 0.5) * step

with open(INPUT) as f:
    rows = list(csv.DictReader(f))

grid = defaultdict(list)

for r in rows:
    chi = float(r["chi"])
    alpha = float(r["peak_alpha_hz"])
    score = float(r["asci"])
    loop = r["candidate_loop"]

    i = bin_index(chi, CHI_MIN, CHI_MAX, CHI_STEP)
    j = bin_index(alpha, ALPHA_MIN, ALPHA_MAX, ALPHA_STEP)
    if i is None or j is None:
        continue
    grid[(i, j)].append((score, loop))

out_rows = []
for (i, j), vals in sorted(grid.items()):
    scores = [v[0] for v in vals]
    loops = [v[1] for v in vals]
    dom = max(set(loops), key=loops.count)
    out_rows.append({
        "chi_center": round(bin_center(i, CHI_MIN, CHI_STEP), 4),
        "alpha_center_hz": round(bin_center(j, ALPHA_MIN, ALPHA_STEP), 4),
        "n": len(vals),
        "mean_score": round(statistics.mean(scores), 6),
        "dominant_loop": dom,
    })

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
    w.writeheader()
    w.writerows(out_rows)

top_bins = sorted(out_rows, key=lambda r: (r["mean_score"], r["n"]), reverse=True)[:15]

with open(OUT_TXT, "w") as f:
    f.write("TOP BINS:\n")
    for r in top_bins:
        f.write(
            f"chi={r['chi_center']}, alpha={r['alpha_center_hz']}, "
            f"mean_score={r['mean_score']}, n={r['n']}, dom_loop={r['dominant_loop']}\n"
        )

print("Saved:")
print("- regime_map.csv")
print("- regime_map.txt")
