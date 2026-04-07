import csv
import statistics
import argparse
from collections import defaultdict

CHI_MIN, CHI_MAX, CHI_STEP = 0.20, 0.80, 0.02
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = 0.5, 14.0, 0.25


def bin_index(x, lo, hi, step):
    if x < lo or x >= hi:
        return None
    return int((x - lo) // step)


def bin_center(idx, lo, step):
    return lo + (idx + 0.5) * step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="diversified_train.csv")
    ap.add_argument("--out-csv", default="regime_map.csv")
    ap.add_argument("--out-txt", default="regime_map.txt")
    args = ap.parse_args()

    with open(args.input) as f:
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

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)

    top_bins = sorted(out_rows, key=lambda r: (r["mean_score"], r["n"]), reverse=True)[:15]
    with open(args.out_txt, "w") as f:
        f.write("TOP BINS:\n")
        for r in top_bins:
            f.write(
                f"chi={r['chi_center']}, alpha={r['alpha_center_hz']}, mean_score={r['mean_score']}, n={r['n']}, dom_loop={r['dominant_loop']}\n"
            )

    print("Saved:")
    print(f"- {args.out_csv}")
    print(f"- {args.out_txt}")


if __name__ == "__main__":
    main()
