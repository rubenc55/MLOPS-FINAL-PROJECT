# Shipping Cost Prediction

> Racing Cargo should fund cost prediction as an MLOps project: tested, weekly retrained estimates replace generic tables, cutting quote time and winning more customers.

This repository turns the exploration notebook into a tested, configurable batch pipeline. One command, `python -m src.main`, cleans and validates the shipments, trains a `RandomForestRegressor` that predicts `cost_usd`, releases the model only if `mape` on the validation split is at most 0.18, and writes an estimated cost for every new shipment. Every setting lives in [`config.yaml`](config.yaml).

## 1. Business case

### Context
- **Client:** Racing Cargo, a freight forwarder. Users: Sales and Pricing teams.
- **Maturity:** data MEDIUM (all shipments, quotes and won or lost outcomes are recorded in a MySQL database fed by an in-house TMS). Tools: MySQL, in-house TMS, Python models, Power BI. Processes: documented quoting flow (customer -> sales rep -> Pricing -> carriers). People: Business Analytics codes in Python and adopts Git from this project.
- **Owners:** Business Analytics retrains and monitors the model. IT maintains the infrastructure and the TMS.

### Problem and baseline
Sales reps and the Pricing team struggle with slow quote responses because each quote is priced individually with every carrier, so some customers choose a competitor. Today the customer asks a sales rep, the rep asks Pricing, and Pricing quotes with carriers.

- Quote response time: 24 to 48 h per quote (team estimate, to be measured in the TMS as quote delivery date minus request date)
- Hit rate: 20% (quotes won / total quotes, from the TMS)

### Goal and success metrics
| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Quote response time | 24 to 48 h (estimate) | At most 2 h, within 3 months of go-live |
| Business KPI | Hit rate | 20% | 35%, within 3 months of go-live |
| ML metric (reported) | `mae` | To be measured | Reported on every run; not a release condition |
| ML metric (quality gate) | `mape` | To be measured | At most 18% on unseen data |

- **Link:** a low `mae` lets sales reps trust the estimated cost and quote without waiting for Pricing, which cuts response time and raises the hit rate.
- **Release rule:** if `mape` on validation exceeds 18%, the model is not released. `mae` is reported on every run but does not block release (decision 2026-11-09: costs range from hundreds to tens of thousands of USD, so one absolute limit does not fit every shipment; first real run on validation: mae 274 USD, mape 16.6%).
- **Limits:** under 5 min per estimate; cost per prediction negligible on current infrastructure (estimate).

### Solution
When a sales rep enters a quote request, the Pricing analyst sees the model's estimated cost in USD, reviews it and shares it with Sales. The sales rep adds the margin and sends the customer a price within 2 h. Pricing confirms the cost with the carrier afterwards.

- **Prediction:** `cost_usd` from transport mode, cargo type, distance, weight, volume, urgency and month, in real time (on request).
- **Fallback:** if the model is unavailable or the shipment is outside the estimation parameters, the system notifies it and Pricing confirms costs directly with carriers.

### Data
- **Source:** `shipments.xlsx`, a synthetic extract that simulates the TMS (MySQL). Owner: IT (TMS); Business Analytics (training data). Size: 5,000 shipments, current year.
- **Label:** `cost_usd`, the carrier cost. Assumption: quoted cost equals invoiced cost.
- **Inputs:** all known at quoting time. Assumption: declared weight and volume equal actual values.
- **Refresh:** new shipments arrive daily; the model is retrained weekly.
- **Personal or sensitive data:** No. The dataset is synthetic and has no customer data.

### Value, cost and risks
- **Benefit:** cost estimates in minutes instead of 24 to 48 h, using more variables than a generic tariff table (accuracy advantage not yet measured).
- **Rough cost (estimate):** Business Analytics team, about 8,000 USD per month in salaries for 3 months (about 24,000 USD); existing IT infrastructure; 12 weeks to production.
- **Scales to:** all company branches, routes and services.

| Risk | Mitigation |
| --- | --- |
| Carrier-confirmed cost differs from the price already sent to the customer | If the gap exceeds 18% or 200 USD, Pricing analyses the root cause and evaluates switching to a carrier with prices closer to the estimate |
| Business Analytics has no Git experience | All code is managed in Git from this project onwards |
| Poorly captured data in the TMS (wrong units, typos) | Data cleaning in the SQL database and data validation in the pipeline |
| Market costs change (for example fuel prices) | Weekly retraining to capture price changes |

### Open gaps
- Benefit over a simple tariff table (for example median cost per transport mode) is not measured.
- Current quote response time is an estimate; to be measured in the TMS.
- Maturity levels for tools, processes and people are not rated.
- Rules that define a shipment "outside the estimation parameters" are not defined.

## 2. Setup

Requires conda (Anaconda or Miniconda). From the repo root:

```bash
conda env create -f environment.yml
conda activate shipping-cost
```

This installs Python 3.12 with pandas, numpy, scikit-learn, joblib, pyyaml, pytest, pytest-cov and flake8. If you created the environment before `pyyaml` was added, update it with `conda env update -f environment.yml --prune`. Run every command below from the repo root with the environment active.

## 3. Data

Nothing under `data/` or `models/` is committed, not even the folders (see `.gitignore`), so a fresh clone has no `data/` folder. Before the first run, create the two input folders from the repo root:

```bash
mkdir -p data/raw data/inference      # macOS, Linux or Git Bash
mkdir data/raw, data/inference        # Windows PowerShell
```

Then put the input files in them:

| File | Contents | Where it comes from |
| --- | --- | --- |
| `data/raw/shipments.csv` | Training data, one row per shipment: `shipment_id`, `transport_mode`, `cargo_type`, `distance_km`, `weight_kg`, `volume_m3`, `urgent`, `month`, `cost_usd` | Sheet `data` of `shipments.xlsx` (section 1), exported once as CSV (UTF-8) |
| `data/inference/new_shipments.csv` | New shipments to price: the same columns without `cost_usd` | The shipments that need a cost estimate |

The output folders (`data/processed/`, `models/` and `reports/`) are created by the run itself. To read the inputs from another location, change `paths.raw_data` and `paths.inference_data` in `config.yaml`.

## 4. Run the pipeline

```bash
python -m src.main
```

The run loads and cleans the raw shipments, validates them against the rules in `config.yaml`, splits them 70/15/15 into train, validation and test, fits the feature recipe and the model on train only, applies the quality gate on validation, reports the test metrics, and then prices the new shipments with the same cleaning and rules. Progress is logged to the terminal, and the run ends with `Done. Model: ... | Predictions: ...`.

It writes three files, none of them committed:

| Output | Default path (`config.yaml` key) | Written when |
| --- | --- | --- |
| Clean training table | `data/processed/clean.csv` (`paths.clean_data`) | On every run, after cleaning and validation |
| Trained model (one Pipeline: `preprocess` + `model`) | `models/model.joblib` (`paths.model`) | Only if `mape` on validation is at most 0.18 |
| Predictions: `shipment_id` and `prediction` (cost in USD) per new shipment | `reports/predictions.csv` (`paths.predictions`) | After the model is released |

If the gate fails, the run stops with `Model not released, quality gate failed: ...` and writes no model and no predictions. If the data breaks a rule, it stops with `Data validation failed:` followed by every problem found.

## 5. Test and check style

```bash
pytest --cov=src --cov-branch --cov-report=term-missing
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
```

- `tests/` has one file per module (`test_<module>.py`) with unit tests and at least one integration test. `tests/test_main.py` also runs `main()` end to end on temporary files, so the real `data/`, `models/` and `reports/` folders are not touched.
- Course gate: every test passes, none is skipped, and branch coverage is at least 90% (currently 100%). flake8 prints nothing.

## 6. Configuration

[`config.yaml`](config.yaml) is the single source of every non-secret setting. `src/main.py` reads it once with `load_config` into `SETTINGS` and turns the relative paths into absolute paths under the repo root. To change a setting, edit `config.yaml` and rerun; no code changes are needed. Secrets never go here: they belong in `.env`, whose names are listed in `.env.example`.

| Section | What it sets | Current values |
| --- | --- | --- |
| `paths` | Input and output files, relative to the repo root | See sections 3 and 4 |
| `problem` | Target, problem type and ID column | `cost_usd`, `regression`, `shipment_id` |
| `features` | Categorical (one-hot encoded) and numeric (passed through) columns | `transport_mode`, `cargo_type` / `distance_km`, `weight_kg`, `volume_m3`, `urgent`, `month` |
| `validation` | Allowed categories and numeric ranges, limits included | For example `transport_mode`: Road, Sea, Air; `month`: 1 to 12 |
| `split` | Test and validation shares of the whole dataset, seed, stratification | 0.15, 0.15, 42, off |
| `model_params` | `RandomForestRegressor` settings | `n_estimators` 200, `max_depth` none, `max_features` 0.5, `min_samples_leaf` 5 |
| `quality_gate` | Release limit per metric (lower is better) | `mape: 0.18` |

**Release rule** (the same in section 1, `TEAM_BRIEF.md`, `config.yaml` and `check_quality_gate` in `src/main.py`): the model that predicts `cost_usd` is released only if `mape` on the validation split is at most 0.18 (18%). `mae` is reported on every run but does not block the release.

## 7. Project structure

### The pipeline at a glance

```text
                           main.py  (orchestrator: python -m src.main, settings from config.yaml)
                              |
 data/raw/ -> load_data -> clean_data -> validate -> split -> features -> train -> evaluate
                                            |   (train/val/test) (recipe)   (fit)   (val split)
                                            v                                           |
                                data/processed/clean.csv          models/model.joblib <-+
                                                                           |
 data/inference/ -> load_data -> clean_data -> validate -> infer <---------+
                                                             |
                                                             v
                                                  reports/predictions.csv
```

### Folders and files

```text
.
├── README.md                 # business case, setup, run, test, configuration
├── TEAM_BRIEF.md             # team decisions and module owners
├── config.yaml               # every non-secret setting (paths, columns, rules, split, model, gate)
├── environment.yml           # conda environment: Python 3.12 and packages
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets, logs and generated files out of Git
├── .flake8                   # style checker settings
├── pytest.ini                # test runner settings
├── LICENSE
├── .github/
│   └── pull_request_template.md
├── notebooks/                # the "lab": exploration only (MLOps_Cost_Prediction.ipynb)
├── src/                      # the "factory": one job per module, one logger per module
│   ├── __init__.py           # makes src a package
│   ├── load_data.py          # 1. read raw data
│   ├── clean_data.py         # 2. stateless cleaning (clean_dataframe)
│   ├── validate.py           # 3. data quality gate
│   ├── features.py           # 4. unfitted feature recipe
│   ├── train.py              # 5. fit the pipeline
│   ├── evaluate.py           # 6. metrics on unseen data
│   ├── infer.py              # 7. predictions on new data
│   ├── utils.py              # shared read and write helpers
│   └── main.py               # 0. orchestrator: load_config, then python -m src.main
├── data/
│   ├── raw/                  # shipments.csv (added by you, not in Git)
│   ├── inference/            # new_shipments.csv (added by you, not in Git)
│   └── processed/            # clean.csv (generated, not in Git)
├── models/                   # model.joblib (generated, not in Git)
├── reports/                  # predictions.csv (generated, not in Git)
└── tests/                    # test_<module>.py per module: unit and integration tests
```

## 8. Team and module owners

From [`TEAM_BRIEF.md`](TEAM_BRIEF.md). Business Analytics retrains and monitors the model; IT maintains the infrastructure and the TMS.

| Module | Owner | Branch |
| --- | --- | --- |
| `utils.py` | Ruben Cuellar (@rubenc55) | `feature/utils` |
| `load_data.py` | Ruben Cuellar (@rubenc55) | `feature/load-data` |
| `clean_data.py` | Ruben Cuellar (@rubenc55) | `feature/clean-data` |
| `validate.py` | Ruben Cuellar (@rubenc55) | `feature/validate` |
| `features.py` | Juan (@JELONDONOZ) | `feature/features` |
| `train.py` | Surya (@jyothikasurya2002) | `feature/train` |
| `evaluate.py` | Enrique (@puigenrique) | `feature/evaluate` |
| `infer.py` | Victoria (@victoriasesilva) | `feature/infer` |
| `main.py` | Ruben Cuellar (@rubenc55) | `feature/main` |