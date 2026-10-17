# Temporal driving diffusion

This folder contains the temporal mixed-type diffusion model, schema/codec,
connected-window training, road-coordinate transforms and generation code.
Feature attention alternates with temporal attention. The four bundled
checkpoints contain the trained younger/older generators at seeds101 and202.

Run from the repository root:

```bash
python -m synthetic_generation generate \
  --checkpoint reproduction_inputs/diffusion/ordinary_s101.pt \
  --source-group ordinary --count 1 --seed 2026 --device cpu \
  --output outputs/generated_ordinary
```

New draws are new synthetic examples and need not reproduce the selected24
trajectories. The selected release trajectories are distributed separately.
`--sampling-steps` overrides the checkpoint's number of sampling steps.

For a training demonstration, first export the provided synthetic tuples:

```bash
python main.py preprocess --output outputs/prepared
python -m synthetic_generation train-diffusion \
  --data-dir outputs/prepared/generator/ordinary --source-group ordinary \
  --schema synthetic_generation/configs/driving.json \
  --steps 100 --device cpu --output outputs/diffusion_demo
```

This fits the final road-increment, progress-conditioned architecture on synthetic
inputs. Road, increment and progress metadata are reused from the bundled cohort
generator checkpoint; the new network is fitted on the synthetic training partition.
It does not retrain the original
generator on the unavailable hospital data. The bundled generator configuration
is stored inside each inference checkpoint.

The public figure renderer reads only bundled synthetic plotting arrays and
published paper-panel images. It does not load the original driving records.

The vendored TabDiff noise schedules, their pinned commit, MIT license and
attribution are in `third_party/tabdiff/`.
