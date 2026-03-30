import csv
import random
import math
import statistics
import argparse
from collections import defaultdict

def safe_float(x, d=0.0):
    try:
        return float(x)
    except:
        return d

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

def f_chi(chi, chi0=0.42, sigma=0.08):
    return math.exp(-((chi - chi0) ** 2) / (2.0 * sigma * sigma))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))

LOOPS = ["ct_cingulate", "dmn", "thalamo_cortical", "fronto_parietal"]

LOOP_BASES = {
    "ct_cingulate": {
        "L_eff_m": 0.025, "v_eff_m_per_s": 7.0, "tau_eff_s": 0.0085,
        "pl_local_aw": 0.82, "pl_global_aw": 0.88, "cfc": 0.72,
        "w_prop": 0.90, "s_struct": 0.90, "peak_alpha_hz": 10.0,
    },
    "dmn": {
        "L_eff_m": 0.040, "v_eff_m_per_s": 6.2, "tau_eff_s": 0.0130,
        "pl_local_aw": 0.60, "pl_global_aw": 0.74, "cfc": 0.58,
        "w_prop": 0.67, "s_struct": 0.78, "peak_alpha_hz": 8.8,
    },
    "thalamo_cortical": {
        "L_eff_m": 0.030, "v_eff_m_per_s": 8.0, "tau_eff_s": 0.0080,
        "pl_local_aw": 0.74, "pl_global_aw": 0.83, "cfc": 0.68,
        "w_prop": 0.84, "s_struct": 0.86, "peak_alpha_hz": 9.6,
    },
    "fronto_parietal": {
        "L_eff_m": 0.035, "v_eff_m_per_s": 6.8, "tau_eff_s": 0.0105,
        "pl_local_aw": 0.66, "pl_global_aw": 0.78, "cfc": 0.62,
        "w_prop": 0.76, "s_struct": 0.82, "peak_alpha_hz": 9.1,
    },
}

GROUP_EFFECTS = {
    "control": {"behavior": 0.90},
    "mci": {"behavior": 0.63},
    "ad_mild": {"behavior": 0.42},
}

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def jitter(x, frac):
    return x * random.uniform(1.0 - frac, 1.0 + frac)

def augment_rows(seed_rows, n=120):
    out = []
    for i in range(n):
        base = random.choice(seed_rows)
        loop = random.choice(LOOPS)
        lb = LOOP_BASES[loop]
        group = base.get("group", "control")
        ge = GROUP_EFFECTS.get(group, GROUP_EFFECTS["control"])

        r = dict(base)
        r["subject_id"] = f'{base.get("subject_id","S")}_aug{i}'
        r["candidate_loop"] = loop

        r["L_eff_m"] = str(jitter(lb["L_eff_m"], 0.12))
        r["v_eff_m_per_s"] = str(jitter(lb["v_eff_m_per_s"], 0.10))
        r["tau_eff_s"] = str(jitter(lb["tau_eff_s"], 0.12))

        r["pl_local_aw"] = str(clamp(jitter(lb["pl_local_aw"], 0.12), 0.05, 0.99))
        r["pl_global_aw"] = str(clamp(jitter(lb["pl_global_aw"], 0.12), 0.05, 0.99))
        r["cfc"] = str(clamp(jitter(lb["cfc"], 0.15), 0.05, 0.99))
        r["w_prop"] = str(clamp(jitter(lb["w_prop"], 0.12), 0.05, 0.99))
        r["s_struct"] = str(clamp(jitter(lb["s_struct"], 0.10), 0.05, 0.99))
        r["peak_alpha_hz"] = str(max(0.5, jitter(lb["peak_alpha_hz"], 0.08)))

        behavior = ge["behavior"]
        if loop == "ct_cingulate":
            behavior += 0.04
        elif loop == "thalamo_cortical":
            behavior += 0.02
        behavior = clamp(jitter(behavior, 0.08), 0.05, 0.999)
        r["behavior_awareness_score"] = str(behavior)

        out.append(r)

    return seed_rows + out

def analyze(rows, chi0, sigma, weights, cingulate_penalty=1.0):
    w_fchi, w_pl, w_pg, w_cfc, w_prop, w_struct = weights
    out = []

    for r in rows:
        rr = dict(r)

        L = safe_float(rr.get("L_eff_m"), 0.03)
        v = safe_float(rr.get("v_eff_m_per_s"), 7.0)
        tau = safe_float(rr.get("tau_eff_s"), 0.01)
        chi = L / max(1e-9, (v * tau))
        fchi = f_chi(chi, chi0, sigma)

        pl_local = safe_float(rr.get("pl_local_aw"))
        pl_global = safe_float(rr.get("pl_global_aw"))
        cfc = safe_float(rr.get("cfc"))
        wprop = safe_float(rr.get("w_prop"))
        sstruct = safe_float(rr.get("s_struct"))
        alpha = safe_float(rr.get("peak_alpha_hz"), 9.5)

        coherence = cfc * pl_global
        phase_disrupt = abs(pl_local - pl_global)

        # gentle slow-wave support, not enough to hijack the model
        delta_component = max(0.0, 4.0 - alpha) / 4.0
        low_chi_bias = max(0.0, 0.6 - chi)
        slow_wave_score = delta_component * low_chi_bias

        asci = (
            w_fchi * fchi
            + w_pl * pl_local
            + w_pg * pl_global
            + w_cfc * cfc
            + w_prop * wprop
            + w_struct * sstruct
            + 0.10 * coherence
            - 0.20 * phase_disrupt
            + 0.12 * slow_wave_score + 0.18 * slow_wave_score * (1.0 - coherence) * max(0.0, 0.6 - chi)
        )

        asci *= random.uniform(0.95, 1.05)

        if rr.get("candidate_loop") == "ct_cingulate":
            asci *= cingulate_penalty

        rr["chi"] = chi
        rr["asci"] = asci
        out.append(rr)

    return out

def summarize_by_loop(rows):
    loops = defaultdict(list)
    for r in rows:
        loops[r.get("candidate_loop", "unknown")].append(r)

    out = []
    for k, rs in loops.items():
        out.append({
            "candidate_loop": k,
            "n": len(rs),
            "mean_asci": statistics.mean(float(r["asci"]) for r in rs),
            "mean_chi": statistics.mean(safe_float(r.get("chi", 0.0)) for r in rs),
            "mean_alpha_hz": statistics.mean(safe_float(r.get("peak_alpha_hz", 0.0)) for r in rs),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", default="out")
    ap.add_argument("--chi0", type=float, default=0.42)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--aug-n", type=int, default=120)
    ap.add_argument("--cingulate-penalty", type=float, default=1.0)
    args = ap.parse_args()

    random.seed(42)
    weights = [0.22, 0.18, 0.18, 0.14, 0.16, 0.12]

    rows = read_rows(args.input)
    rows = augment_rows(rows, n=args.aug_n)

    random.shuffle(rows)
    split = int(0.7 * len(rows))
    train = rows[:split]
    test = rows[split:]

    train_scored = analyze(train, args.chi0, args.sigma, weights, args.cingulate_penalty)
    test_scored = analyze(test, args.chi0, args.sigma, weights, args.cingulate_penalty)

    r_train = pearson(
        [float(r["asci"]) for r in train_scored],
        [safe_float(r.get("behavior_awareness_score")) for r in train_scored],
    )
    r_test = pearson(
        [float(r["asci"]) for r in test_scored],
        [safe_float(r.get("behavior_awareness_score")) for r in test_scored],
    )

    print("=== MODEL VALIDATION ===")
    print(f"TRAIN corr: {r_train:.4f}")
    print(f"TEST  corr: {r_test:.4f}")

    write_csv(f"{args.out_prefix}_train.csv", train_scored)
    write_csv(f"{args.out_prefix}_test.csv", test_scored)
    write_csv(f"{args.out_prefix}_by_loop.csv", summarize_by_loop(train_scored))

if __name__ == "__main__":
    main()
