# Awareness Modeling

Adaptive neural regime modeling framework for mapping synthetic and real EEG-like dynamics into functional brain-state attractors using coherence, phase, and slow-wave scoring.

## What this repo does

This project explores whether compact mesoscale features can separate wake-like, NREM-like, and REM-like regimes while tracking dominant loop families such as:

- `thalamo_cortical`
- `fronto_parietal`
- `dmn`
- `ct_cingulate`

The current pipeline supports:

- synthetic augmentation and scoring
- regime-map generation in chi/alpha space
- heuristic sleep-state projection
- scoring of real EEG/MRI-style summary rows

## Files

- `asci_pipeline_nompl.py` — synthetic augmentation, scoring, train/test split
- `regime_map_nompl.py` — binning and regime-map generation
- `sleep_projection.py` — wake/N1-N2/N3/REM projection from regime bins
- `real_data_ingest_nompl.py` — scoring for real or literature-derived summary rows
- `asci_template.csv` — starter synthetic input template
- `realdata_template.csv` — starter real-data template

## Quick start

```bash
python asci_pipeline_nompl.py --input asci_template.csv --out-prefix diversified
python regime_map_nompl.py
python sleep_projection.py
python real_data_ingest_nompl.py --input realdata_template.csv --out-prefix pilot_real
```

## Current behavior

The model currently tends to produce:

- a coordinated wake-like basin
- a lower-scoring N1/N2-like transitional regime
- a more structured REM-like regime
- incomplete or weak N3 / slow-wave basin capture

That last point is an active modeling target, not a settled result.

## Data format

Expected columns for ingest/scoring include:

- `subject_id`
- `state`
- `group`
- `candidate_loop`
- `L_eff_m`
- `v_eff_m_per_s`
- `tau_eff_s`
- `pl_local_aw`
- `pl_global_aw`
- `cfc`
- `w_prop`
- `s_struct`
- `peak_alpha_hz`
- `behavior_awareness_score`

## Disclaimer

This repository is experimental research software. It is not intended for diagnosis, treatment, anesthesia, or any medical decision-making.
