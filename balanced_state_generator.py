import csv
import random

INPUT = "asci_template.csv"
OUTPUT = "balanced_states.csv"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def jitter(x, frac):
    return x * random.uniform(1.0 - frac, 1.0 + frac)


def make_row(base, idx, state):
    r = dict(base)
    r["subject_id"] = f"{base.get('subject_id','S')}_{state}_{idx}"
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
        r["candidate_loop"] = random.choice(["dmn", "fronto_parietal", "thalamo_cortical"])
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
        r["candidate_loop"] = random.choice(["dmn", "ct_cingulate"])
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
    return r


def main():
    random.seed(42)
    rows = read_rows(INPUT)
    lookup = {}
    for state in ["wake", "n2", "rem", "n3"]:
        match = next((r for r in rows if str(r.get("state", "")).lower() == state), rows[0])
        lookup[state] = match
    out = []
    for i in range(120):
        for state in ["wake", "n2", "rem", "n3"]:
            out.append(make_row(lookup[state], i, state))
    write_rows(OUTPUT, out)
    print(f"Wrote {OUTPUT} with {len(out)} rows")


if __name__ == "__main__":
    main()
