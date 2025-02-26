# Evaluation Pipeline

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate vc-evaluation
```

## Initialization

Configure appropriately the configuration file in `configs/<configuration-of-evaluation>.yml`

## Usage

Run the pipeline by providing the YAML config file:

```bash
python evaluate.py --config configs/example-dice.yml
```

## Development
Place your metric modules (e.g., `dice.py`) in the `metrics` folder.