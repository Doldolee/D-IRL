# Running the public workflows

Use Python3.11+ and run commands from this repository's root. Training and
inference work on CPU; the diffusion and IRL training commands also accept CUDA.

```bash
python -m pip install -r requirements.txt
python main.py check
```

The release contains24 synthetic trajectories with232,525 transitions. Each has
51-dimensional fixed-model states, three bounded actions, next states, DQI
components, scenario labels and one final terminal flag. The generator-frame
states are retained for the frozen behavior-density model and diffusion adapter.
The manifest assigns 10 training and 2 validation trajectories per cohort
for optional new-model training demonstrations. Frozen TA and recovered-reward
evaluation use all 24 trajectories; the reported OPE evaluates each policy on
all 12 corresponding-cohort trajectories, regardless of this demonstration split.
This split was not used to train the bundled original-data policies or density BCs.
It is separate from the original study's train/test split.
The ODA/YDA IRL and comparison BC weights trained on the original data are
provided separately in `reproduction_inputs/policies/`.

## Preprocessing

```bash
python main.py preprocess --output outputs/prepared
```

This checks continuity, reward totals, input hashes and exact coordinate
conversion, then exports model-frame and generator-frame tuple arrays.
It does not read hospital records or refit the frozen transforms.

## IRL and BC training demonstrations

```bash
python main.py train --group senior --steps 20000 --output outputs/irl_senior
python main.py train-bc --group ordinary --epochs 10 --output outputs/bc_ordinary
python main.py train --group ordinary --steps 20000 \
  --bc-init outputs/bc_ordinary/policy.pth --output outputs/irl_ordinary
```

The IRL code keeps the architecture, update equations and cohort warm-start
setting. No initialization-only BC checkpoints are bundled: the ordinary-group
example explicitly initializes from the BC trained on the public synthetic data.
These commands train on synthetic data and produce new checkpoints;
they do not regenerate the released policies trained on the original data.
Use smaller `--steps` or `--epochs` for a CPU smoke run. Outputs never overwrite
the bundled models.

## Frozen-reference trajectory diagnostics

```bash
python main.py ta --output outputs/ta
python main.py ta --recover --output outputs/ta_with_recovery
```

`--limit 1` is available for a smoke run. These commands infer critic values and
policy/kernel densities from the released trajectories and frozen networks.
They do not read stored TA scores as inputs. Saved scores are provided only as
comparison values in `reproduction_inputs/expected_ta.json`.
TA-dSNIPS is conditional on the supplied frozen synthetic behavior-density
models. It is an observed-state diagnostic, not a sequential-return estimator.

## OPE

The README reports the completed original-policy/frozen-evaluator comparison on
synthetic data. The exact published table values are available as JSON:

```bash
python main.py ope-results
```

DR, K-WIS and independent FQE mathematics are in `ope.py`. To demonstrate a new
FQE fit and validation evaluation using synthetic inputs:

```bash
python main.py fit-fqe --policy-key senior_auto_p2 --iterations 600 --updates 50 \
  --output outputs/fqe_senior
python main.py evaluate-fqe --policy-key senior_auto_p2 \
  --checkpoint outputs/fqe_senior/evaluator_iteration600.pth \
  --output outputs/fqe_senior_validation
```

A new synthetic FQE fit is a new demonstration and will not reproduce the README
table, which used frozen evaluators trained on the unavailable original data.
Those original evaluator files and individual original evaluation arrays are not
included in this release.

## Figures and diffusion

```bash
python -m synthetic_generation.figures --output outputs/figures
python -m synthetic_generation.overview --output outputs/architecture
```

The first command renders four comparison figures from stored plot inputs,
including an existing t-SNE embedding. It does not refit t-SNE or recompute scores.
The architecture diagram is generated as SVG and PNG. Diffusion generation and
training commands are documented in `synthetic_generation/README.md`.

Original per-record driving data, private run histories, CSV exports and remote
execution settings are not included. `RELEASE_MANIFEST.json` lists the intended
public files and their hashes; files created under `outputs/` are not uploaded.
