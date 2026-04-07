import csv
import math
import statistics
from collections import Counter

INPUT = "balanced_states.csv"
OUT_CSV = "recurrent_rollout_map.csv"
OUT_TXT = "recurrent_rollout_map.txt"

CHI_MIN, CHI_MAX, CHI_STEP = 0.20, 0.80, 0.04
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = 0.5, 14.0, 0.5
ROLLOUT_STEPS = 8

WAKE_INERTIA = 0.18
REM_INERTIA = 0.10
N2_INERTIA = 0.08
N3_INERTIA = 0.28

LOOP_STICKINESS = {"thalamo_cortical": 0.03, "fronto_parietal": 0.03, "dmn": 0.02, "ct_cingulate": 0.05}
TRANSITION_PENALTY = {("wake", "n2"): 0.02, ("n2", "wake"): 0.015, ("n2", "n3"): 0.01, ("n3", "n2"): 0.03, ("wake", "rem"): 0.015, ("rem", "wake"): 0.02}
FEATURES = ["peak_alpha_hz", "L_eff_m", "v_eff_m_per_s", "tau_eff_s", "pl_local_aw", "pl_global_aw", "cfc", "w_prop", "s_struct"]
LOOPS = ["thalamo_cortical", "fronto_parietal", "dmn", "ct_cingulate"]
STATES = ["wake", "n2", "rem", "n3"]


def safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def f_chi(chi, chi0=0.42, sigma=0.08):
    return math.exp(-((chi - chi0) ** 2) / (2.0 * sigma * sigma))


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_row(rows):
    out = {}
    for k in FEATURES + ["behavior_awareness_score"]:
        vals = [safe_float(r.get(k)) for r in rows]
        out[k] = statistics.mean(vals)
    return out


def zscore_refs(refs):
    stats = {}
    for k in FEATURES:
        vals = [refs[s][k] for s in refs]
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        stats[k] = (mu, sd)
    return stats


def dist_to_state(row, ref, norm_stats):
    total = 0.0
    for k in FEATURES:
        mu, sd = norm_stats[k]
        x = safe_float(row.get(k))
        a = (x - mu) / sd
        b = (ref[k] - mu) / sd
        total += (a - b) ** 2
    return total ** 0.5


def classify_state(row, refs, norm_stats):
    best_state = None
    best_dist = None
    for s, ref in refs.items():
        d = dist_to_state(row, ref, norm_stats)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_state = s
    return best_state, best_dist


def state_inertia_gain(state_label):
    return {"wake": WAKE_INERTIA, "rem": REM_INERTIA, "n2": N2_INERTIA, "n3": N3_INERTIA}.get(state_label, 0.0)


def transition_cost(prev_state, curr_state):
    if prev_state is None or curr_state is None or prev_state == curr_state:
        return 0.0
    return TRANSITION_PENALTY.get((prev_state, curr_state), 0.015)


def base_score(row):
    weights = [0.22, 0.18, 0.18, 0.14, 0.16, 0.12]
    w_fchi, w_pl, w_pg, w_cfc, w_prop, w_struct = weights
    L = safe_float(row["L_eff_m"])
    v = safe_float(row["v_eff_m_per_s"])
    tau = safe_float(row["tau_eff_s"])
    chi = L / max(1e-9, v * tau)
    pl_local = safe_float(row["pl_local_aw"])
    pl_global = safe_float(row["pl_global_aw"])
    cfc = safe_float(row["cfc"])
    wprop = safe_float(row["w_prop"])
    sstruct = safe_float(row["s_struct"])
    alpha = safe_float(row["peak_alpha_hz"])
    coherence = cfc * pl_global
    phase_disrupt = abs(pl_local - pl_global)
    delta_component = max(0.0, 4.0 - alpha) / 4.0
    low_chi_bias = max(0.0, 0.6 - chi)
    slow_wave_score = delta_component * low_chi_bias
    asci = (
        w_fchi * f_chi(chi)
        + w_pl * pl_local
        + w_pg * pl_global
        + w_cfc * cfc
        + w_prop * wprop
        + w_struct * sstruct
        + 0.10 * coherence
        - 0.20 * phase_disrupt
        + 0.12 * slow_wave_score
        + 0.18 * slow_wave_score * (1.0 - coherence) * max(0.0, 0.6 - chi)
    )
    rem_window = max(0.0, 1.0 - abs(alpha - 8.0) / 2.5) * max(0.0, 1.0 - abs(chi - 0.50) / 0.12)
    if row["candidate_loop"] == "thalamo_cortical":
        asci += 0.05 * rem_window
    if row["candidate_loop"] == "ct_cingulate":
        asci *= 0.9
    return asci


def blend(a, b, t):
    return a * (1.0 - t) + b * t


def make_target_row(ref, alpha_target, chi_target):
    row = dict(ref)
    tau = safe_float(ref["tau_eff_s"])
    v = safe_float(ref["v_eff_m_per_s"])
    row["peak_alpha_hz"] = alpha_target
    row["L_eff_m"] = chi_target * v * tau
    return row


def rollout_to_target(start_state, alpha_target, chi_target, refs, norm_stats):
    start = dict(refs[start_state])
    prev_state = start_state
    prev_loop = {"wake": "thalamo_cortical", "n2": "dmn", "rem": "thalamo_cortical", "n3": "ct_cingulate"}[start_state]
    prev_asci = None
    current = dict(start)
    visited = []
    for step in range(ROLLOUT_STEPS):
        t = (step + 1) / ROLLOUT_STEPS
        target_ref = make_target_row(refs[start_state], alpha_target, chi_target)
        probe = {}
        for k in FEATURES + ["behavior_awareness_score"]:
            probe[k] = blend(safe_float(current[k]), safe_float(target_ref[k]), t)
        best = None
        for loop in LOOPS:
            cand = dict(probe)
            cand["candidate_loop"] = loop
            state_label, _ = classify_state(cand, refs, norm_stats)
            asci = base_score(cand)
            if prev_asci is not None:
                asci += state_inertia_gain(prev_state) * prev_asci
            if loop == prev_loop:
                asci += LOOP_STICKINESS.get(loop, 0.02)
            asci -= transition_cost(prev_state, state_label)
            item = {"loop": loop, "state_label": state_label, "asci_mem": asci, "row": cand}
            if best is None or item["asci_mem"] > best["asci_mem"]:
                best = item
        current = dict(best["row"])
        prev_state = best["state_label"]
        prev_loop = best["loop"]
        prev_asci = best["asci_mem"]
        visited.append(best)
    final = visited[-1]
    return {"final_state": final["state_label"], "final_loop": final["loop"], "final_score": round(final["asci_mem"], 6)}


def bin_centers(lo, hi, step):
    vals = []
    n = int((hi - lo) / step)
    for i in range(n):
        vals.append(lo + (i + 0.5) * step)
    return vals


rows = read_rows(INPUT)
refs = {}
for state in STATES:
    rs = [r for r in rows if str(r.get("state", "")).lower() == state]
    refs[state] = mean_row(rs)

norm_stats = zscore_refs(refs)
alphas = bin_centers(ALPHA_MIN, ALPHA_MAX, ALPHA_STEP)
chis = bin_centers(CHI_MIN, CHI_MAX, CHI_STEP)

out_rows = []
for prev_state in STATES:
    for alpha in alphas:
        for chi in chis:
            result = rollout_to_target(prev_state, alpha, chi, refs, norm_stats)
            out_rows.append({
                "prev_state": prev_state,
                "alpha_center_hz": round(alpha, 4),
                "chi_center": round(chi, 4),
                "final_state": result["final_state"],
                "final_loop": result["final_loop"],
                "final_score": result["final_score"],
            })

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
    w.writeheader()
    w.writerows(out_rows)

with open(OUT_TXT, "w") as f:
    for prev_state in STATES:
        sub = [r for r in out_rows if r["prev_state"] == prev_state]
        counts = Counter(r["final_state"] for r in sub)
        loops = Counter(r["final_loop"] for r in sub)
        f.write(f"=== prev_state={prev_state} ===\n")
        f.write(f"state_counts={dict(counts)}\n")
        f.write(f"loop_counts={dict(loops)}\n\n")

print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_TXT}")
for prev_state in STATES:
    sub = [r for r in out_rows if r["prev_state"] == prev_state]
    counts = Counter(r["final_state"] for r in sub)
    loops = Counter(r["final_loop"] for r in sub)
    print(f"\nprev_state={prev_state}")
    print("  state_counts:", dict(counts))
    print("  loop_counts :", dict(loops))
