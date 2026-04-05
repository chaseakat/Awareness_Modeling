# Four-State Recurrent Milestone

This milestone captures the first version of the awareness-state modeling pipeline that shows all of the following at once:

- separable **wake**, **N2**, **REM**, and **N3** regimes
- a stable synthetic generator that can keep N3 alive instead of starving it out
- path-based transition tests that distinguish wake, N2, REM, and N3 trajectories
- state-specific inertia that breaks forward/reverse symmetry
- recurrent rollout maps whose final-state occupancy depends strongly on the prior state

## Milestone result

The most important result in this milestone is the recurrent rollout map.

Observed endpoint distributions from the rollout experiment:

- **start from wake** → mostly wake, some REM, almost no N2
- **start from N2** → mostly N2, with a substantial REM branch
- **start from REM** → mostly REM, some N2, very little wake
- **start from N3** → almost entirely N3, with negligible escape to N2

This is the first point in the project where the model behaves more like a simple recurrent regime system than a static score table.

## Interpretation

A compact summary of the current architecture:

- **Wake** behaves like a strongly self-preserving coordinated basin.
- **N2** behaves like a transitional basin that can feed REM.
- **REM** behaves like a metastable side-basin rather than just a weaker wake state.
- **N3** behaves like the deepest and most self-preserving basin in the current model family.

## Remaining weakness

The main remaining weakness is REM internal structure.

At this milestone, REM is dynamically distinct, but its dominant loop identity is still too simple and still overlaps too much with wake-like coordinated structure. The next phase after this milestone should focus on **REM loop diversification** and improved internal composition.

## Files introduced in this milestone

- `balanced_state_generator.py`
- `asci_pipeline_nompl.py`
- `regime_map_nompl.py`
- `sleep_projection.py`
- `real_data_ingest_nompl.py`
- `state_specific_inertia.py`
- `recurrent_rollout_map.py`
- `asci_template.csv`
- `realdata_template.csv`

## Warning

This repository is experimental research software. None of the scores, labels, or attractor claims in this repo should be treated as medical inference, clinical diagnosis, anesthesia guidance, or a validated neuroscience model.
