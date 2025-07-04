# Network Intrusion Detection System (NIDS)

A Python package and FastAPI service for detecting malicious network flows using a Random-Forest classifier trained on the CIC-IDS2017 dataset.

## Quickstart

```bash
# 1. Install dependencies (preferably inside a virtualenv)
python -m pip install -r requirements.txt

# 2. Train model (reads CSVs in data/raw by default)
python -m nids.pipelines.train --data-dir data/raw

# 3. Run predictions
python -m nids.pipelines.predict my_flows.csv

# 4. Launch API (reloads model on-demand)
uvicorn app.main:app --reload
```

## Repository Layout

```
├── nids/                # Importable source code
│   ├── pipelines/       # Training & inference pipelines
│   ├── models/          # Stand-alone model experiments
│   └── utils/           # Helper scripts (e.g. CSV repair)
├── app/                 # FastAPI service
├── data/
│   ├── raw/             # Original CIC-IDS2017 CSVs (git-ignored)
│   └── processed/       # Cleaned / re-packed subsets (git-ignored)
├── trained_model_files/ # Saved model & encoder artefacts
├── reports/             # Evaluation reports + plots
└── tests/               # Pytest suite
```

## Dataset Preparation

Place the raw CIC-IDS2017 CSVs under `data/raw/`. If your files contain broken rows, repair them first:

```bash
python -m nids.utils.repair_cic_ids_csv data/raw data/processed
```

Then train the model with `--data-dir data/processed`.

## Development

- **Linting**: `ruff check .`
- **Tests**: `pytest -q`
- **Docker**: `docker build -t nids-api . && docker run -p 8000:8000 nids-api`

CI runs linting, unit tests and a minimal Docker health-check on every push.

---

Made with ❤ and open-source tools.
