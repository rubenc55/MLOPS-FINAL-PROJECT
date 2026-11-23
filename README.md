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
| ML metric (reported) | `mae` | 892 USD (no-ML table, validation) | Reported on every run; not a release condition |
| ML metric (quality gate) | `mape` | 56.0% (no-ML table, validation) | At most 18% on unseen data |

- **Link:** a low `mae` lets sales reps trust the estimated cost and quote without waiting for Pricing, which cuts response time and raises the hit rate.
- **Release rule:** if `mape` on validation exceeds 18%, the model is not released. `mae` is reported on every run but does not block release (decision 2026-11-09: costs range from hundreds to tens of thousands of USD, so one absolute limit does not fit every shipment; first real run on validation: mae 274 USD, mape 16.6%).
- **Baseline:** the no-ML table is the median train cost per transport mode, the best of three simple tables we tried (`evaluate_baseline` in `src/evaluate.py`, `baseline` in `config.yaml`). Every run logs it next to the model, so the comparison is repeated with each retraining.
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
- **Benefit:** cost estimates in minutes instead of 24 to 48 h. On the same unseen shipments the model is about 3 times more accurate than a tariff table (measured 2026-11-20, train 3,500 / validation 750 / test 750 shipments):

| Method | Validation `mape` | Validation `mae` | Test `mape` | Test `mae` |
| --- | --- | --- | --- | --- |
| One median cost for every shipment | 88.5% | 1,046 USD | 96.3% | 917 USD |
| Median cost per transport mode (baseline) | 56.0% | 892 USD | 59.8% | 756 USD |
| Rate per km by mode and cargo type, urgent premium | 69.7% | 860 USD | 63.5% | 700 USD |
| Model (`RandomForestRegressor`) | 16.5% | 274 USD | 16.0% | 215 USD |

73.5% of the model's validation estimates are within 18% of the carrier cost, against 16.9% for the per-km table.
- **Rough cost (estimate):** Business Analytics team, about 8,000 USD per month in salaries for 3 months (about 24,000 USD); existing IT infrastructure; 12 weeks to production.
- **MLOps running cost:** 0 USD per month at the current volume, on the free plans of GitHub Actions (CI and deploy), DagsHub (data, and MLflow runs and model registry) and Render (API, sleeps after 15 minutes without requests). A paid Render plan is needed only when quotes cannot wait about a minute for the service to wake up (check each provider's pricing page before scaling).
- **When it is not worth it:** for a branch with few quotes, or if carriers publish fixed tariffs, the median table above is cheaper to run and good enough.
- **Scales to:** all company branches, routes and services.

| Risk | Mitigation |
| --- | --- |
| Carrier-confirmed cost differs from the price already sent to the customer | If the gap exceeds 18% or 200 USD, Pricing analyses the root cause and evaluates switching to a carrier with prices closer to the estimate |
| Business Analytics has no Git experience | All code is managed in Git from this project onwards |
| Poorly captured data in the TMS (wrong units, typos) | Data cleaning in the SQL database and data validation in the pipeline |
| Market costs change (for example fuel prices) | Weekly retraining to capture price changes |
| A retrained model is worse than the one in use | Quality gate (`mape` at most 0.18 on validation): a failing model is not saved, registered or deployed |
| A broken release reaches Pricing | CI on every pull request, release only from `main`, smoke test of the live service, rollback to the previous tag in minutes (section 7) |

### Open gaps
- Current quote response time is an estimate; to be measured in the TMS.
- Maturity levels for tools, processes and people are not rated.
- Rules that define a shipment "outside the estimation parameters" are not defined.

## 2. Setup

Requires conda (Anaconda or Miniconda). From the repo root:

```bash
conda env create -f environment.yml
conda activate shipping-cost
```

For the exact versions of CI and Docker instead, install the lock: `conda-lock install -n shipping-cost conda-lock.yml`.

This installs Python 3.12 with pandas, numpy, scikit-learn, joblib, pyyaml, mlflow-skinny, python-dotenv, fastapi, pydantic, uvicorn, pytest, pytest-cov, flake8, dvc, httpx and conda-lock. If you created the environment before `mlflow-skinny` was added, update it with `conda env update -f environment.yml --prune`. Run every command below from the repo root with the environment active.

Secrets go in a `.env` file at the repo root, which Git ignores. Copy the names from `.env.example` and fill in your own values:

```bash
cp .env.example .env      # then set MLFLOW_TRACKING_USERNAME (DagsHub user) and MLFLOW_TRACKING_PASSWORD (DagsHub token)
```

Without `MLFLOW_TRACKING_PASSWORD` the pipeline still runs; it only skips MLflow tracking (section 4). The token is the same kind as for `dvc pull` (DagsHub > Your Settings > Tokens).

## 3. Data

The CSV files are never committed. Git keeps only small DVC pointers (`data/raw/shipments.csv.dvc` and `data/inference/new_shipments.csv.dvc`) with the hash of each file; the files themselves live in the project's DagsHub storage ([jyothikasurya2002/MLOPS-FINAL-PROJECT](https://dagshub.com/jyothikasurya2002/MLOPS-FINAL-PROJECT), public). DagsHub asks for a login even on public repos, so before the first `dvc pull` save your DagsHub user and token (DagsHub > Your Settings > Tokens) on your machine only:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-user>
dvc remote modify origin --local password <dagshub-token>
dvc pull
```

`--local` writes to `.dvc/config.local`, which Git ignores, so the token is never committed.

| File | Contents | Where it comes from |
| --- | --- | --- |
| `data/raw/shipments.csv` | Training data, one row per shipment: `shipment_id`, `transport_mode`, `cargo_type`, `distance_km`, `weight_kg`, `volume_m3`, `urgent`, `month`, `cost_usd` | Sheet `data` of `shipments.xlsx` (section 1), exported once as CSV (UTF-8) |
| `data/inference/new_shipments.csv` | New shipments to price: the same columns without `cost_usd` | Stand-in until the TMS export exists: the first 20 shipments of sheet `data`, without `cost_usd` |

The output folders (`data/processed/`, `models/`, `reports/` and `logs/`) are created by the run itself. To read the inputs from another location, change `paths.raw_data` and `paths.inference_data` in `config.yaml` and in `dvc.yaml`.

To replace a data file, overwrite it, then run `dvc add <file>` and `dvc push`, and commit the updated `.dvc` pointer. `dvc push` needs write access to the DagsHub repo (ask the owner to add you as a collaborator).

## 4. Run the pipeline

```bash
dvc repro
```

`dvc repro` runs the only stage in `dvc.yaml`, `python -m src.main`, and records the hash of the code, data, settings and every output in `dvc.lock`. If nothing changed since the last run, it skips the run. Commit `dvc.lock` after every run that changes it, and send the new model to DagsHub with `dvc push`. `python -m src.main` alone runs the same pipeline without updating `dvc.lock`.

The run loads and cleans the raw shipments, validates them against the rules in `config.yaml`, splits them 70/15/15 into train, validation and test, fits the feature recipe and the model on train only, applies the quality gate on validation, reports the test metrics, and then prices the new shipments with the same cleaning and rules. Progress is logged to the terminal and appended to `logs/pipeline.log`, and the run ends with `Done. Model: ... | Predictions: ...`.

It writes four files, none of them committed:

| Output | Default path (`config.yaml` key) | Written when |
| --- | --- | --- |
| Clean training table | `data/processed/clean.csv` (`paths.clean_data`) | On every run, after cleaning and validation |
| Trained model (one Pipeline: `preprocess` + `model`) | `models/model.joblib` (`paths.model`) | Only if `mape` on validation is at most 0.18 |
| Predictions: `shipment_id` and `prediction` (cost in USD) per new shipment | `reports/predictions.csv` (`paths.predictions`) | After the model is released |
| Run log: every step, the validation and test metrics and the gate decision | `logs/pipeline.log` (`paths.log_file`) | On every run, appended (also when the gate or validation fails); not a DVC output, so its timestamps never change `dvc.lock` |

Every run first deletes the model and predictions of the previous run. If the gate fails, the run stops with `Model not released, quality gate failed: ...` (also written to `logs/pipeline.log` as an ERROR), so `models/` and `reports/` stay empty and no model rejected by the gate, or older than it, can be deployed. If the data breaks a rule, it stops with `Data validation failed:` followed by every problem found.

### Tracking and model registry

`src/main.py` owns one MLflow run per pipeline run, on the MLflow server of the project's DagsHub repo (experiment `shipping-cost`, `tracking` in `config.yaml`; open it in DagsHub > Experiments). Each run records:

| What | Where in MLflow |
| --- | --- |
| Every setting of `config.yaml` | Run parameters (for example `model_params.n_estimators`) and the artifact `run_config.json` |
| The md5 of `data/raw/shipments.csv` (the same hash as in `shipments.csv.dvc`) and the Git commit | Run tags `data_md5`, `git_commit` |
| `val_mae`, `val_mape`, `baseline_val_mae`, `baseline_val_mape`, and after the gate `test_mae`, `test_mape` | Run metrics |
| Why a run stopped (failed gate or bad data) | Run status FAILED, tag `error` |
| The model, only if the gate passes and the predictions are written | Registered model `shipping_cost_model`: a new version with tags `data_md5` and `git_commit` (a candidate; `prod` does not move) |

Any version leads back to its run, data version and commit. Without `MLFLOW_TRACKING_PASSWORD` (or with `tracking.mode: disabled`) the run logs a warning and registers nothing. CI never has the MLflow login, so only a run on a laptop with `.env` filled in registers a version.

### Promote a model to prod

Passing the gate makes a model a candidate. A person decides which candidate ships: the next release builds its image with the version that has the alias `prod`, and moving `prod` changes nothing that is already live.

1. Train once with `.env` filled in: `dvc repro --force`. The log ends with `Model registered as shipping_cost_model version N (candidate)`.
2. Open the MLflow UI at the `tracking_uri` of `config.yaml` (https://dagshub.com/jyothikasurya2002/MLOPS-FINAL-PROJECT.mlflow, DagsHub login). Experiments > `shipping-cost`, tick the new run and the run of the current `prod` version, Compare. Look at validation only: `val_mape` and `val_mae`, lower is better. Promote only if the candidate is better; test metrics are a final audit, never the reason to choose.
3. Models > `shipping_cost_model` > Version N. Check its Source Run is the run you compared, then Aliases > Add `prod` and confirm the move.
4. Models > `shipping_cost_model` shows `prod` on Version N and on no other version. Then commit `dvc.lock`, merge to `main` and publish a release (section 7).

A run tag called prod, renaming a run or picking the newest version is not promotion. `prod` is one label on one version.

## 5. Serve the API

`src/api.py` serves the released model (`models/model.joblib`) over HTTP. It adds no ML logic: every request goes through the same `clean_dataframe`, `validate_dataframe` and `run_inference` as the batch run, so a live estimate equals the batch estimate for the same shipment.

| Endpoint | What it does |
| --- | --- |
| `GET /health` | `status` ok, the release `version`, the code `git_commit` of the image, the `model_version` it serves (for example `shipping_cost_model:v3`; `local` for a model trained locally), the `model` file and `model_info`: MLflow run, training commit, data md5 and metrics of the prod model |
| `POST /predict` | Estimated `cost_usd` per shipment, in the request order |
| `GET /docs` | The contract and a form to try it, with the sample below (the only UI) |

Sample request ([`scripts/sample_request.json`](scripts/sample_request.json) holds three; the smoke test and CI send that file):

```json
{"records": [{"shipment_id": 1, "transport_mode": "Sea", "cargo_type": "General", "distance_km": 8928.0,
  "weight_kg": 1683.2, "volume_m3": 4.719, "urgent": 0, "month": 3}]}
```

Answer: `{"model_version": "shipping_cost_model:v3", "predictions": [{"shipment_id": 1, "prediction": 1308.73}]}` (cost in USD, with the version of the model that made it).

The request is strict (Pydantic, `extra="forbid"`): an unknown or missing field, a number sent as text, `true` for `urgent` or an empty list gets `422` before the model runs. A shipment that breaks a rule in `config.yaml` (unknown category, value out of range) also gets `422`, with the same message as the batch run. Without a model the service does not start, and the error says to run the pipeline first.

Run it locally, after `dvc repro` has written the model:

```bash
uvicorn src.api:app --port 8000          # then open http://localhost:8000/docs
```

Or in Docker, the same image Render runs:

```bash
docker build -t api .
docker run -p 8000:8000 api              # same URLs as above
```

The image installs the exact versions in `conda-lock.yml` (linux-64, the platform of CI and Render; on Apple chips Docker emulates it). `.dockerignore` is an allow-list: only `src/`, `config.yaml`, `conda-lock.yml` and `models/model.joblib` enter the build, never data, tests, notebooks, `.env` or `.git`. After any change to `environment.yml`, rewrite the lock with `conda-lock lock -f environment.yml -p linux-64 -p osx-arm64 -p osx-64 -p win-64` and commit it.

## 6. Test and check style

```bash
pytest --cov=src --cov-branch --cov-report=term-missing
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
```

- `tests/` has one file per module (`test_<module>.py`) with unit tests and at least one integration test. `tests/test_main.py` also runs `main()` end to end on temporary files, so the real `data/`, `models/` and `reports/` folders are not touched.
- Course gate: every test passes, none is skipped, and branch coverage is at least 90% (currently 100%). flake8 prints nothing.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull request into `dev` or `main` and on every push to them, with the exact packages of `conda-lock.yml`. Each check is its own job, so a failing one shows by name:

| Job | Runs | Fails when |
| --- | --- | --- |
| `lint (flake8)` | `flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests` | flake8 prints anything |
| `test (pytest, 90% branch coverage)` | `pytest --cov=src --cov-branch --cov-fail-under=90` | a test fails or branch coverage is under 90% |
| `pipeline (dvc repro, quality gate)` | `dvc pull`, then `dvc status` must be clean, then `dvc repro --force` | `dvc.lock` is stale, data validation fails or the gate is not met |
| `docker (build, /health, /predict)` | `docker build -t api .` with the model of the pipeline job, then calls the running container | the image does not build, or `/health`, `/predict` or the 422 check fail |

With branch protection on `dev` and `main` (Settings > Branches: require these four checks to pass), a red check blocks the merge. For `dvc pull` the pipeline job reads two repository settings, `DAGSHUB_USER` and `DAGSHUB_TOKEN`. It does not get the MLflow login, so CI trains only to prove the code still works: the model is thrown away, nothing is registered and `prod` never moves. The run log is kept as the `pipeline-log` artifact of each run.

## 7. Release, deploy and roll back

Nothing reaches the live service without a person publishing a GitHub Release from `main`; [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) does the rest:

| Step | Job | Stops the release when |
| --- | --- | --- |
| 1 | `release guard (tag on main)` ([`release-guard.yml`](.github/workflows/release-guard.yml)) | the tag is not on `main` |
| 2 | CI again on the tagged commit (all four jobs of `ci.yml`) | any check is red |
| 3 | `build release image (prod model)`: `python -m scripts.fetch_prod_model` downloads `shipping_cost_model@prod` from MLflow on DagsHub with its lineage (`models/model_info.json`), then `docker build --build-arg APP_VERSION=<tag>` | there is no prod model or the image does not build |
| 4 | Smoke test of the image, then push to `ghcr.io/<owner>/<repo>:<tag>` | `scripts/smoke_test.py` fails |
| 5 | `deploy to Render and smoke test`: the Render deploy hook gets the image of this tag, then `scripts/smoke_test.py` waits until the live `/health` reports the tag | the live service does not answer, or answers with the old tag |

To release:

1. If the release ships a new model, promote it first (section 4). Merge into `main` through pull requests, with CI green.
2. On GitHub: Releases > Draft a new release, new tag `vX.Y.Z` on target `main`, notes from the changelog at the end of this README, Publish.
3. Watch Actions > Deploy. When it is green, `GET <RENDER_URL>/health` shows the new `version` and, in `model_info`, the model version, MLflow run, Git commit and data md5 of the served model.

Check the live service by hand at any time (Render free services sleep after 15 minutes and wake on the first call):

```bash
python scripts/smoke_test.py --url <RENDER_URL>
```

**Roll back:** every release keeps its own image, so going back does not rebuild anything. Actions > Deploy > Run workflow, enter the older tag (for example `v1.0.0`), Run. The guard checks the tag is on `main`, Render deploys that image and the smoke test waits until `/health` reports it. Code and model go back together, because the image holds both. If the older release serves another model version, move `prod` back to that version too (section 4, step 3), so the next release does not ship the model you rolled back from. To return to the latest release, move `prod` forward first, then run it again with the latest tag.

| Date | From | To | Why | Result |
| --- | --- | --- | --- | --- |
| 2026-11-22 20:13 UTC | `v1.0.1` | `v1.0.0` | Rollback practice | Green in 82 s ([run](https://github.com/rubenc55/MLOPS-FINAL-PROJECT/actions/runs/45065094741)); `/health` reported `v1.0.0` |
| 2026-11-22 20:19 UTC | `v1.0.0` | `v1.0.1` | Back to the latest release | Green in 82 s ([run](https://github.com/rubenc55/MLOPS-FINAL-PROJECT/actions/runs/45065629385)); `/health` reported `v1.0.1` |

Settings the repo owner adds once (Settings > Secrets and variables > Actions): secrets `DAGSHUB_USER`, `DAGSHUB_TOKEN` (DVC and MLflow), `RENDER_DEPLOY_HOOK` (Render > service > Settings > Deploy Hook) and variable `RENDER_URL` (`https://<service>.onrender.com`). The Render service is an image-backed web service that pulls `ghcr.io/<owner>/<repo>:<tag>`, with Health Check Path `/health`. The deploy hook only changes the tag, so the service must already point to that image: when creating a new service, set Settings > Image to `ghcr.io/<owner>/<repo>:<first tag>` once, then every release and rollback goes through the hook.

## 8. Configuration

[`config.yaml`](config.yaml) is the single source of every non-secret setting. `src/config.py` reads it once with `load_config` into `SETTINGS` and turns the relative paths into absolute paths under the repo root; `src/main.py` (and later the API) imports `SETTINGS` from there. To change a setting, edit `config.yaml` and rerun; no code changes are needed. Secrets never go here: they belong in `.env`, whose names are listed in `.env.example`.

| Section | What it sets | Current values |
| --- | --- | --- |
| `paths` | Input and output files, relative to the repo root | See sections 3 and 4 |
| `logging` | Log level and line format for the console and `paths.log_file` | `INFO`, `time \| level \| module \| message` |
| `problem` | Target, problem type and ID column | `cost_usd`, `regression`, `shipment_id` |
| `features` | Categorical (one-hot encoded) and numeric (passed through) columns | `transport_mode`, `cargo_type` / `distance_km`, `weight_kg`, `volume_m3`, `urgent`, `month` |
| `validation` | Allowed categories and numeric ranges, limits included | For example `transport_mode`: Road, Sea, Air; `month`: 1 to 12 |
| `split` | Test and validation shares of the whole dataset, seed, stratification | 0.15, 0.15, 42, off |
| `model_params` | `RandomForestRegressor` settings | `n_estimators` 200, `max_depth` none, `max_features` 0.5, `min_samples_leaf` 5 |
| `quality_gate` | Release limit per metric (lower is better) | `mape: 0.18` |
| `tracking` | Tracking mode, MLflow server, experiment, registered model name and the alias of the released model | `online`, the DagsHub repo's MLflow, `shipping-cost`, `shipping_cost_model`, `prod` |

**Release rule** (the same in section 1, `TEAM_BRIEF.md`, `config.yaml` and `check_quality_gate` in `src/main.py`): the model that predicts `cost_usd` is released only if `mape` on the validation split is at most 0.18 (18%). `mae` is reported on every run but does not block the release.

## 9. Project structure

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
├── conda-lock.yml            # exact versions of every package (linux-64, osx-arm64, osx-64, win-64)
├── Dockerfile                # serving image: locked packages, src/, config.yaml and the model
├── .dockerignore             # allow-list of what the Docker build may see
├── dvc.yaml                  # DVC stage: python -m src.main, its inputs and outputs
├── dvc.lock                  # hashes of the last run (generated by dvc repro, committed)
├── .dvc/                     # DVC settings: config (DagsHub remote) is committed, config.local is not
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets, logs and generated files out of Git
├── .flake8                   # style checker settings
├── pytest.ini                # test runner settings
├── LICENSE
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml            # CI: lint, test, pipeline with gate, Docker build on every pull request
│       ├── deploy.yml        # CD: a published release (or a rollback tag) goes live on Render
│       └── release-guard.yml # stops a release whose tag is not on main
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
│   ├── config.py             # load_config: config.yaml -> SETTINGS
│   ├── logger.py             # setup_logging: console + logs/pipeline.log
│   ├── tracking.py           # MLflow run, metrics and prod model (called by main)
│   ├── api.py                # FastAPI: GET /health, POST /predict, /docs
│   └── main.py               # 0. orchestrator: python -m src.main
├── data/
│   ├── raw/                  # shipments.csv.dvc in Git; shipments.csv from dvc pull
│   ├── inference/            # new_shipments.csv.dvc in Git; new_shipments.csv from dvc pull
│   └── processed/            # clean.csv (generated, not in Git)
├── models/                   # model.joblib (generated, not in Git)
├── reports/                  # predictions.csv (generated, not in Git)
├── logs/                     # pipeline.log (generated, not in Git)
├── scripts/
│   ├── fetch_prod_model.py   # download the MLflow prod model and its lineage (python -m scripts.fetch_prod_model)
│   ├── sample_request.json   # sample body for POST /predict, used by the smoke test and CI
│   └── smoke_test.py         # check a running API: python scripts/smoke_test.py --url <url>
└── tests/                    # test_<module>.py per module: unit and integration tests
```

## 10. Team and module owners

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

## 11. Model card

| Item | Value |
| --- | --- |
| Model | `RandomForestRegressor` (`n_estimators` 200, `max_features` 0.5, `min_samples_leaf` 5, seed 42) in one scikit-learn Pipeline with the one-hot encoder of `features.py` |
| Predicts | `cost_usd`, the carrier cost in USD of one shipment, for Pricing to review before Sales adds the margin |
| Inputs | `transport_mode`, `cargo_type`, `distance_km`, `weight_kg`, `volume_m3`, `urgent`, `month`, all known at quoting time; `shipment_id` only identifies the row |
| Training data | `data/raw/shipments.csv` (5,000 synthetic shipments, no personal data), version in `data/raw/shipments.csv.dvc`; split 70/15/15, fitted on train only |
| Quality gate | `mape` at most 0.18 on validation, otherwise nothing is saved, registered or deployed |
| Metrics (2026-11-20) | Validation `mape` 16.5%, `mae` 274 USD; test `mape` 16.0%, `mae` 215 USD; no-ML table: validation `mape` 56.0% (section 1) |
| Valid range | The rules in `config.yaml` (`validation`): known modes and cargo types, distance 50 to 12,000 km, weight 10 to 9,900 kg, volume 0.028 to 93.213 m3, month 1 to 12. Shipments outside them get `422` and go to Pricing by hand |
| Limits | Least accurate on the most expensive quarter of shipments (above about 830 USD): validation `mape` 20.7% against about 15% for the other three quarters, so Pricing confirms those estimates with the carrier first; trained on one synthetic year, so seasonality and fuel prices beyond it are not learned; assumes declared weight and volume are the real ones |
| Trace a prediction | Every `/predict` answer names its `model_version`; `GET /health` adds the release `version`, the code `git_commit`, and in `model_info` the MLflow run, training commit and data md5 of that model |
| Owner | Business Analytics retrains weekly and monitors; IT runs the infrastructure |

## 12. Changelog

One entry per release tag; the notes of each GitHub Release copy its entry.

### v1.0.1 (2026-11-22)
- Same code and prod model as v1.0.0 (`shipping_cost_model` version 3), rebuilt and deployed as its own release
- Used to practise the rollback: v1.0.1 -> v1.0.0 -> v1.0.1 (section 7)

### v1.0.0 (2026-11-22)
- Settings in `config.yaml` read by `src/config.py`; one logger to the console and `logs/pipeline.log`
- Data versioned with DVC: pointers and `dvc.lock` in Git, data on DagsHub; `dvc repro` runs `python -m src.main`
- Quality gate enforced: `mape` at most 0.18 on validation, older outputs removed, failures logged
- MLflow runs on DagsHub per pipeline run; the model that passes the gate is registered as `shipping_cost_model` with alias `prod`
- FastAPI service (`/health`, `/predict`, `/docs`) in a Docker image built from `conda-lock.yml`
- CI on every pull request: flake8, tests at 90% branch coverage, pipeline with the gate, Docker build
- Release and deploy to Render from GitHub Releases, smoke test, rollback by tag
- No-ML baseline (median cost per transport mode) scored in every run as evidence for the business case
