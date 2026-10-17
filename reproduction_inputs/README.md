# Reproduction inputs

The 24 synthetic trajectories contain 232,525 transitions: 12 younger-group and 12 older-group drives. No original hospital records are included.

- `synthetic24/`: states, next states, actions, DQI rewards, scenario labels and terminal flags. Model-frame and generator-frame states are supplied separately.
- `policies/`: the original-data-trained fixed ODA/YDA IRL and comparison BC models. Both IRL models use seed 2 and 20,000 steps.
- `behavior/`: eight frozen synthetic behavior-density models used by TA-dSNIPS.
- `diffusion/`: younger/older temporal diffusion models for seeds 101 and 202.
- `transforms.json`: state-coordinate transforms used by the fixed models.
- `expected_ta.json`: saved reference scores for checking new TA evaluations.
- `ope_results.json`: the reported OPE table values.

`data.py` retains the cohort assignments, random seeds and optional synthetic training split (10 training and 2 validation drives per cohort). New training examples fit new synthetic-data models; the original-data-trained policies are supplied separately. New diffusion samples may differ from the released trajectories.
