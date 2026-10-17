# Reproduction inputs

`manifest.json` identifies24 synthetic trajectories, their group, a synthetic
20train/4validation demonstration split, random-draw seeds, behavior-model
assignment and file hashes. No original hospital trajectory arrays or participant
identifiers are included.

The train/validation labels support optional new-model training demonstrations
only (10/2 trajectories per cohort). Frozen TA and recovered-reward evaluation
use all24; reported OPE uses all12 corresponding-cohort drives per policy.
These labels do not describe the training data of the bundled fixed models.
The original-data-trained ODA/YDA IRL and comparison BC weights are provided
separately in `policies/`.

- `synthetic24/driveNNN.npz`: fixed-model states and next states, the corresponding
  generator-frame states, three actions, DQI component/total rewards, scenario
  labels and terminal flags. State frames are explicitly separate arrays.
- `policies/irl_*.pth`: the fixed ODA/YDA policy and critic networks, with their
  native entropy temperature and seed2/20k configuration.
- `policies/bc_*.pth`: the two BC policies used in the reported OPE comparison.
- `behavior/`: eight frozen synthetic behavior-density networks. Their inputs
  use the generator-frame states, as specified in each checkpoint.
- `diffusion/`: younger/older temporal diffusion inference checkpoints for
  seeds101 and202. Original training-row indices and run manifests are excluded.
- `transforms.json`: numerical state-coordinate transforms and feature names.
- `expected_ta.json`: saved comparison scores; not used as inference inputs.
- `ope_results.json`: the60 reported policy/estimator/DQI result rows.

Training commands fit new illustrative models on synthetic data. They do not
retrain the original hospital-data policies. New diffusion samples need not
match the selected24 release trajectories. The exact full synthetic dataset is
provided directly for repeatable downstream evaluation.
