# HiddenCourtesyMerge-Sim

Controlled benchmark for belief-based planning under hidden driver courtesy in
highway merge scenarios.

## Environment

The current manuscript and scripts were checked with Python 3.13 and the pinned
packages in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Core Reproduction Commands

Observation-model calibration:

```bash
python calibrate_observation_model.py --episodes 60 --seed 999 --output observation_model.json
```

Dataset generation:

```bash
python generate_hidden_courtesy_merge_dataset.py --episodes 3000 --seed 7 --output HiddenCourtesyMerge-Sim-cleanobs
```

Main heuristic policy evaluation:

```bash
python evaluate_policies.py --episodes 300 --seed 17 --output eval_final_v6_cleanobs --policies belief_policy rule_policy random_policy oracle_policy
```

POMCP evaluation:

```bash
python evaluate_policies.py --episodes 300 --seed 17 --output pomcp_300ep_100sims --policies pomcp_policy --pomcp-sims 100 --pomcp-horizon 10
```

POMCP rollout-isolation evaluation:

```bash
python evaluate_policies.py --episodes 300 --seed 17 --output pomcp_heuristic_rollout_300ep --policies pomcp_policy --pomcp-sims 100 --pomcp-horizon 10 --pomcp-rollout-d-thresh 8 --pomcp-rollout-ttc-thresh 3
```

Belief-gain sweep:

```bash
python run_belief_gain_sweep.py --episodes 300 --seed 17 --output belief_gain_sweep --gains 0.05 0.10 0.185 0.30 0.50
```

Paper build:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Repository Hygiene

Large generated datasets, pilot runs, smoke-test outputs, local installers, and
LaTeX build products are intentionally ignored by `.gitignore`. The paper source,
tables, figures, and scripts are the canonical commit targets. Large generated
data should be released through an anonymous artifact repository or archival
service for review.
