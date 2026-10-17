# Run the examples

Use Python 3.11+ and run commands from the repository root.

## Install and check inputs

```bash
python -m pip install -r requirements.txt
python main.py check
```

## Reproduce the figures

Render the six comparison figures and diffusion overview as PNGs.

```bash
python -m synthetic_generation.figures --output outputs/figures
python -m synthetic_generation.overview --output outputs/architecture
```

## Evaluate the released policies

Compute TA and recovered rewards with the fixed models; print the saved OPE results.

```bash
python main.py ta --recover --output outputs/ta
python main.py ope-results
```

## Optional training examples

Train new models on the synthetic 10/2 train/validation split per cohort. The released original-data-trained weights are separate; these examples do not retrain them.

```bash
python main.py preprocess --output outputs/prepared
python main.py train --group senior --steps 20000 --output outputs/irl_senior
python main.py train-bc --group ordinary --epochs 10 --output outputs/bc_ordinary
python main.py train --group ordinary --steps 20000 \
  --bc-init outputs/bc_ordinary/policy.pth --output outputs/irl_ordinary
```

Diffusion generation and training: [synthetic_generation/README.md](synthetic_generation/README.md).
