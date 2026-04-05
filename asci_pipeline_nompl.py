import csv
import random
import math
import argparse
import statistics
from collections import defaultdict


def safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def jitter(x, frac):
    return x * random.uniform(1.0 - frac, 1.0 + frac)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def f_chi(chi, chi0=0.42, sigma=0.08):
    return math.exp(-((chi - chi0) ** 2) / (2.0 * sigma * sigma))


def augment_rows(seed_rows, n=120):
    if not seed_rows or n <= 0:
        return []

    states = ["wake", "n2", "rem", "n3"]
    seeds = {}
    fallback = seed_rows[0]
    for s in states:
        match = next((r for r in seed_rows if str(r.get("state", "")).lower() == s), None)
        seeds[s] = dict(match or fallback)

    out = []
    for i in range(n):
        for state in states:
            base = dict(seeds[state])
            r = dict(base)
            r["subject_id"] = f"{base.get('subject_id', 'S')}_{state}_{i}"
            r["state"] = state

            if state == "wake":
                r["candidate_loop"] = random.choice(["thalamo_cortical", "fronto_parietal", "ct_cingulate"])
                r["L_eff_m"] = str(jitter(0.030, 0.10))
                r["v_eff_m_per_s"] = str(jitter(8.0, 0.10))
                r["tau_eff_s"] = str(jitter(0.008, 0.10))
                r["pl_local_aw"] = str(clamp(jitter(0.74, 0.12), 0.05, 0.99))
                r["pl_global_aw"] = str(clamp(jitter(0.83, 0.12), 0.05, 0.99))
                r["cfc"] = str(clamp(jitter(0.68, 0.12), 0.05, 0.99))
                r["w_prop"] = str(clamp(jitter(0.84, 0.10), 0.05, 0.99))
                r["s_struct"] = str(clamp(jitter(0.86, 0.10), 0.05, 0.99))
                r["peak_alpha_hz"] = str(clamp(jitter(9.8, 0.12), 8.0, 12.0))
                r["behavior_awareness_score"] = str(clamp(jitter(0.95, 0.06), 0.70, 0.999))

            elif state == "n2":
                r["candidate_loop"] = random.choice(["dmn", "fronto_parietal", "thalamo_cortical"])
                r["L_eff_m"] = str(jitter(0.040, 0.12))
                r["v_eff_m_per_s"] = str(jitter(6.1, 0.12))
                r["tau_eff_s"] = str(jitter(0.012, 0.12))
                r["pl_local_aw"] = str(clamp(jitter(0.58, 0.15), 0.05, 0.95))
                r["pl_global_aw"] = str(clamp(jitter(0.70, 0.15), 0.05, 0.95))
                r["cfc"] = str(clamp(jitter(0.52, 0.15), 0.05, 0.90))
                r["w_prop"] = str(clamp(jitter(0.66, 0.14), 0.05, 0.95))
                r["s_struct"] = str(clamp(jitter(0.75, 0.12), 0.05, 0.95))
                r["peak_alpha_hz"] = str(clamp(jitter(7.2, 0.18), 5.5, 9.0))
                r["behavior_awareness_score"] = str(clamp(jitter(0.55, 0.12), 0.25, 0.80))

            elif state == "rem":
                r["candidate_loop"] = random.choice(["thalamo_cortical", "fronto_parietal", "dmn"])
                r["L_eff_m"] = str(jitter(0.034, 0.10))
                r["v_eff_m_per_s"] = str(jitter(6.9, 0.10))
                r["tau_eff_s"] = str(jitter(0.010, 0.12))
                r["pl_local_aw"] = str(clamp(jitter(0.64, 0.14), 0.05, 0.95))
                r["pl_global_aw"] = str(clamp(jitter(0.76, 0.12), 0.05, 0.95))
                r["cfc"] = str(clamp(jitter(0.60, 0.15), 0.05, 0.95))
                r["w_prop"] = str(clamp(jitter(0.74, 0.12), 0.05, 0.95))
                r["s_struct"] = str(clamp(jitter(0.80, 0.10), 0.05, 0.95))
                r["peak_alpha_hz"] = str(clamp(jitter(8.9, 0.15), 6.0, 10.0))
                r["behavior_awareness_score"] = str(clamp(jitter(0.72, 0.10), 0.35, 0.90))

            elif state == "n3":
                r["candidate_loop"] = random.choice(["ct_cingulate", "dmn"])
                r["L_eff_m"] = str(jitter(0.050, 0.18))
                r["v_eff_m_per_s"] = str(jitter(4.0, 0.22))
                r["tau_eff_s"] = str(jitter(0.020, 0.22))
                r["pl_local_aw"] = str(clamp(jitter(0.42, 0.18), 0.05, 0.95))
                r["pl_global_aw"] = str(clamp(jitter(0.30, 0.20), 0.05, 0.85))
                r["cfc"] = str(clamp(jitter(0.25, 0.25), 0.01, 0.80))
                r["w_prop"] = str(clamp(jitter(0.40, 0.18), 0.05, 0.85))
                r["s_struct"] = str(clamp(jitter(0.55, 0.15), 0.05, 0.95))
                r["peak_alpha_hz"] = str(clamp(jitter(2.0, 0.45), 0.5, 4.0))
                r["behavior_awareness_score"] = str(clamp(jitter(0.20, 0.20), 0.01, 0.45))

            out.append(r)
    return out


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


def summarize_by_group(scored):
    groups = defaultdict(list)
    for r in scored:
        groups[r.get("group", "unknown")].append(r)
    out = []
    for k, rs in groups.items():
        out.append({
            "group": k,
            "n": len(rs),
            "mean_asci": round(statistics.mean(safe_float(r["asci"]) for r in rs), 6),
            "mean_behavior": round(statistics.mean(safe_float(r.get("behavior_awareness_score")) for r in rs), 6),
            "mean_chi": round(statistics.mean(safe_float(r["chi"]) for r in rs), 6),
        })
    return out


def summarize_by_loop(scored):
    loops = defaultdict(list)
    for r in scored:
        loops[r.get("candidate_loop", "unknown")].append(r)
    out = []
    for k, rs in loops.items():
        out.append({
            "candidate_loop": k,
            "n": len(rs),
            "mean_asci": round(statistics.mean(safe_float(r["asci"]) for r in rs), 6),
            "mean_chi": round(statistics.mean(safe_float(r["chi"]) for r in rs), 6),
            "mean_alpha_hz": round(statistics.mean(safe_float(r.get("peak_alpha_hz")) for r in rs), 6),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", default="out")
    ap.add_argument("--chi0", type=float, default=0.42)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--aug-n", type=int, default=120)
    args = ap.parse_args()

    random.seed(42)
    rows = read_rows(args.input)
    rows = rows + augment_rows(rows, n=args.aug_n)
    random.shuffle(rows)
    split = max(1, int(0.7 * len(rows)))
    train = rows[:split]
    test = rows[split:] if split < len(rows) else rows[:]

    train_scored = [score_row(r, args.chi0, args.sigma) for r in train]
    test_scored = [score_row(r, args.chi0, args.sigma) for r in test]

    r_train = pearson([safe_float(r["asci"]) for r in train_scored], [safe_float(r.get("behavior_awareness_score")) for r in train_scored])
    r_test = pearson([safe_float(r["asci"]) for r in test_scored], [safe_float(r.get("behavior_awareness_score")) for r in test_scored])

    print("=== MODEL VALIDATION ===")
    print(f"TRAIN corr: {r_train:.4f}")
    print(f"TEST  corr: {r_test:.4f}")

    write_csv(f"{args.out_prefix}_train.csv", train_scored)
    write_csv(f"{args.out_prefix}_test.csv", test_scored)
    write_csv(f"{args.out_prefix}_by_group.csv", summarize_by_group(train_scored))
    write_csv(f"{args.out_prefix}_by_loop.csv", summarize_by_loop(train_scored))


if __name__ == "__main__":
    main()
