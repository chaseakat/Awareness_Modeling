import csv
import math
import statistics
import argparse
from collections import defaultdict

def safe_float(x, d=0.0):
    try:
        return float(x)
    except:
        return d

def compute_chi(r):
    L = safe_float(r.get("L_eff_m"), 0.0)
    v = safe_float(r.get("v_eff_m_per_s"), 0.0)
    tau = safe_float(r.get("tau_eff_s"), 0.0)
    if v <= 0 or tau <= 0:
        return 0.0
    return L / (v * tau)

def f_chi(chi, chi0=0.42, sigma=0.08):
    return math.exp(-((chi - chi0) ** 2) / (2.0 * sigma * sigma))

def pearson(x, y):
    if len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x) ** 0.5
    deny = sum((b - my) ** 2 for b in y) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)

def score_row(r, weights, chi0=0.42, sigma=0.08):
    w_fchi, w_pl, w_pg, w_cfc, w_prop, w_struct = weights

    chi = compute_chi(r)
    pl_local = safe_float(r.get("pl_local_aw"))
    pl_global = safe_float(r.get("pl_global_aw"))
    cfc = safe_float(r.get("cfc"))
    wprop = safe_float(r.get("w_prop"))
    sstruct = safe_float(r.get("s_struct"))
    alpha = safe_float(r.get("peak_alpha_hz"))

    coherence = cfc * pl_global
    phase_disrupt = abs(pl_local - pl_global)

    delta_component = max(0.0, 4.0 - alpha) / 4.0
    low_chi_bias = max(0.0, 0.6 - chi)
    slow_wave_score = delta_component * low_chi_bias

    asci = (
        w_fchi * f_chi(chi, chi0, sigma)
        + w_pl * pl_local
        + w_pg * pl_global
        + w_cfc * cfc
        + w_prop * wprop
        + w_struct * sstruct
        + 0.10 * coherence
        - 0.20 * phase_disrupt
        + 0.12 * slow_wave_score + 0.18 * slow_wave_score * (1.0 - coherence) * max(0.0, 0.6 - chi)
    )

    return {
        "chi": chi,
        "coherence": coherence,
        "phase_disrupt": phase_disrupt,
        "slow_wave_score": slow_wave_score,
        "asci": asci,
    }

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def summarize(rows, key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(key, "unknown")].append(r)

    out = []
    for k, rs in sorted(buckets.items()):
        out.append({
            key: k,
            "n": len(rs),
            "mean_asci": statistics.mean(safe_float(r.get("asci")) for r in rs),
            "mean_chi": statistics.mean(safe_float(r.get("chi")) for r in rs),
            "mean_alpha_hz": statistics.mean(safe_float(r.get("peak_alpha_hz")) for r in rs),
            "mean_coherence": statistics.mean(safe_float(r.get("coherence")) for r in rs),
            "mean_phase_disrupt": statistics.mean(safe_float(r.get("phase_disrupt")) for r in rs),
            "mean_slow_wave_score": statistics.mean(safe_float(r.get("slow_wave_score")) for r in rs),
        })
    return out

def make_template(path):
    rows = [
        {
            "subject_id": "S001",
            "state": "wake",
            "group": "control",
            "candidate_loop": "thalamo_cortical",
            "L_eff_m": "0.030",
            "v_eff_m_per_s": "8.0",
            "tau_eff_s": "0.008",
            "pl_local_aw": "0.74",
            "pl_global_aw": "0.83",
            "cfc": "0.68",
            "w_prop": "0.84",
            "s_struct": "0.86",
            "peak_alpha_hz": "9.8",
            "behavior_awareness_score": "0.95",
        },
        {
            "subject_id": "S001",
            "state": "n2",
            "group": "control",
            "candidate_loop": "dmn",
            "L_eff_m": "0.040",
            "v_eff_m_per_s": "6.1",
            "tau_eff_s": "0.012",
            "pl_local_aw": "0.58",
            "pl_global_aw": "0.70",
            "cfc": "0.52",
            "w_prop": "0.66",
            "s_struct": "0.75",
            "peak_alpha_hz": "7.2",
            "behavior_awareness_score": "0.55",
        },
        {
            "subject_id": "S001",
            "state": "rem",
            "group": "control",
            "candidate_loop": "fronto_parietal",
            "L_eff_m": "0.034",
            "v_eff_m_per_s": "6.9",
            "tau_eff_s": "0.010",
            "pl_local_aw": "0.64",
            "pl_global_aw": "0.76",
            "cfc": "0.60",
            "w_prop": "0.74",
            "s_struct": "0.80",
            "peak_alpha_hz": "8.9",
            "behavior_awareness_score": "0.72",
        },
        {
            "subject_id": "S001",
            "state": "n3",
            "group": "control",
            "candidate_loop": "dmn",
            "L_eff_m": "0.050",
            "v_eff_m_per_s": "4.0",
            "tau_eff_s": "0.020",
            "pl_local_aw": "0.42",
            "pl_global_aw": "0.30",
            "cfc": "0.25",
            "w_prop": "0.40",
            "s_struct": "0.55",
            "peak_alpha_hz": "2.0",
            "behavior_awareness_score": "0.20",
        },
    ]
    write_csv(path, rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Input CSV")
    ap.add_argument("--out-prefix", default="realdata")
    ap.add_argument("--template", action="store_true", help="Write template CSV and exit")
    ap.add_argument("--template-path", default="realdata_template.csv")
    ap.add_argument("--chi0", type=float, default=0.42)
    ap.add_argument("--sigma", type=float, default=0.08)
    args = ap.parse_args()

    if args.template:
        make_template(args.template_path)
        print(f"Wrote template: {args.template_path}")
        return

    if not args.input:
        raise SystemExit("Provide --input or use --template")

    weights = [0.22, 0.18, 0.18, 0.14, 0.16, 0.12]

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    scored = []
    for r in rows:
        rr = dict(r)
        metrics = score_row(rr, weights, chi0=args.chi0, sigma=args.sigma)
        rr.update(metrics)
        scored.append(rr)

    write_csv(f"{args.out_prefix}_scored.csv", scored)
    state_summary = summarize(scored, "state")
    loop_summary = summarize(scored, "candidate_loop")
    write_csv(f"{args.out_prefix}_by_state.csv", state_summary)
    write_csv(f"{args.out_prefix}_by_loop.csv", loop_summary)

    if all("behavior_awareness_score" in r for r in scored):
        xs = [safe_float(r.get("asci")) for r in scored]
        ys = [safe_float(r.get("behavior_awareness_score")) for r in scored]
        print(f"ASCI vs behavior corr: {pearson(xs, ys):.4f}")

    print("Saved:")
    print(f"- {args.out_prefix}_scored.csv")
    print(f"- {args.out_prefix}_by_state.csv")
    print(f"- {args.out_prefix}_by_loop.csv")

if __name__ == "__main__":
    main()
