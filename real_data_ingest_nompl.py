import csv
import math
import argparse
import statistics
from collections import defaultdict


def safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def f_chi(chi, chi0=0.42, sigma=0.08):
    return math.exp(-((chi - chi0) ** 2) / (2.0 * sigma * sigma))


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x) ** 0.5
    deny = sum((b - my) ** 2 for b in y) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def score_row(rr, chi0=0.42, sigma=0.08):
    L = safe_float(rr.get("L_eff_m"), 0.03)
    v = safe_float(rr.get("v_eff_m_per_s"), 7.0)
    tau = safe_float(rr.get("tau_eff_s"), 0.01)
    chi = L / max(1e-9, v * tau)

    pl_local = safe_float(rr.get("pl_local_aw"))
    pl_global = safe_float(rr.get("pl_global_aw"))
    cfc = safe_float(rr.get("cfc"))
    wprop = safe_float(rr.get("w_prop"))
    sstruct = safe_float(rr.get("s_struct"))
    alpha = safe_float(rr.get("peak_alpha_hz"), 9.5)

    coherence = cfc * pl_global
    phase_disrupt = abs(pl_local - pl_global)
    delta_component = max(0.0, 4.0 - alpha) / 4.0
    low_chi_bias = max(0.0, 0.6 - chi)
    slow_wave_score = delta_component * low_chi_bias

    asci = (
        0.22 * f_chi(chi, chi0, sigma)
        + 0.18 * pl_local
        + 0.18 * pl_global
        + 0.14 * cfc
        + 0.16 * wprop
        + 0.12 * sstruct
        + 0.10 * coherence
        - 0.20 * phase_disrupt
        + 0.12 * slow_wave_score
        + 0.18 * slow_wave_score * (1.0 - coherence) * max(0.0, 0.6 - chi)
    )

    rem_window = (
        max(0.0, 1.0 - abs(alpha - 8.0) / 2.5)
        * max(0.0, 1.0 - abs(chi - 0.50) / 0.12)
    )
    if rr.get("candidate_loop") == "thalamo_cortical":
        asci += 0.05 * rem_window
    if rr.get("candidate_loop") == "ct_cingulate":
        asci *= 0.9

    r = dict(rr)
    r["chi"] = round(chi, 6)
    r["coherence"] = round(coherence, 6)
    r["phase_disrupt"] = round(phase_disrupt, 6)
    r["slow_wave_score"] = round(slow_wave_score, 6)
    r["asci"] = round(asci, 6)
    return r


def summarize(scored, key):
    bucket = defaultdict(list)
    for r in scored:
        bucket[r.get(key, "unknown")].append(r)
    out = []
    for k, rs in bucket.items():
        out.append({
            key: k,
            "n": len(rs),
            "mean_asci": round(statistics.mean(safe_float(r["asci"]) for r in rs), 6),
            "mean_chi": round(statistics.mean(safe_float(r["chi"]) for r in rs), 6),
            "mean_alpha_hz": round(statistics.mean(safe_float(r.get("peak_alpha_hz")) for r in rs), 6),
            "mean_coherence": round(statistics.mean(safe_float(r["coherence"]) for r in rs), 6),
            "mean_phase_disrupt": round(statistics.mean(safe_float(r["phase_disrupt"]) for r in rs), 6),
            "mean_slow_wave_score": round(statistics.mean(safe_float(r["slow_wave_score"]) for r in rs), 6),
        })
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", default="real")
    ap.add_argument("--chi0", type=float, default=0.42)
    ap.add_argument("--sigma", type=float, default=0.08)
    args = ap.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    scored = [score_row(r, chi0=args.chi0, sigma=args.sigma) for r in rows]
    write_csv(f"{args.out_prefix}_scored.csv", scored)
    write_csv(f"{args.out_prefix}_by_state.csv", summarize(scored, "state"))
    write_csv(f"{args.out_prefix}_by_loop.csv", summarize(scored, "candidate_loop"))

    if all("behavior_awareness_score" in r for r in scored):
        xs = [safe_float(r["asci"]) for r in scored]
        ys = [safe_float(r.get("behavior_awareness_score")) for r in scored]
        print(f"ASCI vs behavior corr: {pearson(xs, ys):.4f}")

    print("Saved:")
    print(f"- {args.out_prefix}_scored.csv")
    print(f"- {args.out_prefix}_by_state.csv")
    print(f"- {args.out_prefix}_by_loop.csv")


if __name__ == "__main__":
    main()
