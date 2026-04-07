import csv
import math
import statistics

INPUT = "balanced_states.csv"
OUTPUT = "state_specific_inertia.csv"

WAKE_INERTIA = 0.18
REM_INERTIA = 0.10
N2_INERTIA = 0.08
N3_INERTIA = 0.28

LOOP_STICKINESS = {
    "thalamo_cortical": 0.03,
    "fronto_parietal": 0.03,
    "dmn": 0.02,
    "ct_cingulate": 0.05,
}

TRANSITION_PENALTY = {
    ("wake", "n2"): 0.02,
    ("n2", "wake"): 0.015,
    ("n2", "n3"): 0.01,
    ("n3", "n2"): 0.03,
    ("wake", "rem"): 0.015,
    ("rem", "wake"): 0.02,
}

FEATURES = [
    "peak_alpha_hz",
    "L_eff_m",
    "v_eff_m_per_s",
    "tau_eff_s",
    "pl_local_aw",
    "pl_global_aw",
    "cfc",
    "w_prop",
    "s_struct",
]


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


def blend(a, b, t):
    return a * (1.0 - t) + b * t


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
    return {"wake": WAKE_INERTIA, "rem": REM_INERTIA, "n2": N2_INERTIA, "n3": N3_INERTIA}.get(state_label, 0.10)


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


def transition_cost(prev_state, curr_state):
    if prev_state is None or curr_state is None or prev_state == curr_state:
        return 0.0
    return TRANSITION_PENALTY.get((prev_state, curr_state), 0.015)


def score_with_state_memory(row, refs, norm_stats, prev_asci=None, prev_loop=None, prev_state=None):
    asci = base_score(row)
    state_label, state_dist = classify_state(row, refs, norm_stats)
    if prev_asci is not None:
        asci += state_inertia_gain(prev_state) * prev_asci
    if prev_loop is not None and row["candidate_loop"] == prev_loop:
        asci += LOOP_STICKINESS.get(row["candidate_loop"], 0.02)
    asci -= transition_cost(prev_state, state_label)
    return state_label, state_dist, asci


def make_path(name, a, b, loop, refs, norm_stats, steps=25):
    rows = []
    prev_asci = None
    prev_loop = None
    prev_state = None
    for i in range(steps):
        t = i / (steps - 1)
        row = {"path": name, "step": i, "candidate_loop": loop}
        for k in a:
            row[k] = blend(a[k], b[k], t)
        state_label, state_dist, asci_mem = score_with_state_memory(row, refs, norm_stats, prev_asci, prev_loop, prev_state)
        row["asci"] = round(base_score(row), 6)
        row["asci_mem"] = round(asci_mem, 6)
        row["state_label"] = state_label
        row["state_dist"] = round(state_dist, 6)
        rows.append(row)
        prev_asci = asci_mem
        prev_loop = row["candidate_loop"]
        prev_state = state_label
    return rows


rows = read_rows(INPUT)
refs = {}
for state in ["wake", "n2", "rem", "n3"]:
    rs = [r for r in rows if str(r.get("state", "")).lower() == state]
    refs[state] = mean_row(rs)

norm_stats = zscore_refs(refs)
paths = []
paths += make_path("wake_to_n2", refs["wake"], refs["n2"], "thalamo_cortical", refs, norm_stats)
paths += make_path("n2_to_n3", refs["n2"], refs["n3"], "ct_cingulate", refs, norm_stats)
paths += make_path("n3_to_n2", refs["n3"], refs["n2"], "ct_cingulate", refs, norm_stats)
paths += make_path("n2_to_wake", refs["n2"], refs["wake"], "thalamo_cortical", refs, norm_stats)
paths += make_path("wake_to_rem", refs["wake"], refs["rem"], "thalamo_cortical", refs, norm_stats)
paths += make_path("rem_to_wake", refs["rem"], refs["wake"], "thalamo_cortical", refs, norm_stats)

with open(OUTPUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=paths[0].keys())
    w.writeheader()
    w.writerows(paths)

print(f"Saved: {OUTPUT}")
for name in sorted(set(r["path"] for r in paths)):
    rs = [r for r in paths if r["path"] == name]
    raw = [float(r["asci"]) for r in rs]
    mem = [float(r["asci_mem"]) for r in rs]
    labels = [r["state_label"] for r in rs]
    print(f"\n{name}")
    print(f"  raw start/end: {raw[0]:.4f} {raw[-1]:.4f}")
    print(f"  mem start/end: {mem[0]:.4f} {mem[-1]:.4f}")
    print(f"  raw min/max: {min(raw):.4f} {max(raw):.4f}")
    print(f"  mem min/max: {min(mem):.4f} {max(mem):.4f}")
    print(f"  first/mid/last label: {labels[0]} {labels[len(labels)//2]} {labels[-1]}")
