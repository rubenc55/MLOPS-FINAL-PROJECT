# Team brief

## Business
- Business case: README.md, section 1 (merged into dev on 2026-11-01)
- Use case: Pricing analysts decide the reference cost shared with Sales using the predicted cost_usd
- Notebook: notebooks/MLOps_Cost_Prediction.ipynb
- Target: cost_usd | Problem type: regression | ID column: shipment_id

## Pipeline
- Settings: every value below (paths, target, features, validation rules, split, model, quality gate) lives in config.yaml at the repo root; src/config.py loads it with load_config into SETTINGS, and src/logger.py sends logs to the console and logs/pipeline.log. Change settings there, not in code
- Feature groups: categorical: transport_mode, cargo_type -> OneHotEncoder(handle_unknown="ignore") (text, 3 values each)
- Feature groups: numeric: distance_km, weight_kg, volume_m3, urgent, month -> passthrough (no scaling for trees)
- Excluded columns: shipment_id (identifier, kept next to each prediction for traceability)
- Model: RandomForestRegressor with n_estimators=200, max_depth=None, max_features=0.5, min_samples_leaf=5
- Metric: mae and mape, keys in evaluate_model output: ["mae", "mape"], quality gate: mape <= 0.18 (mae reported, not gated)
- Split: test 0.15 and validation 0.15, both of the full dataset; random_state 42
- Validation rules: required columns; no missing values; cost_usd > 0 (training only)
- Validation rules: transport_mode in Road/Sea/Air; cargo_type in General/Refrigerated/Hazardous; urgent 0/1; month 1-12
- Validation rules: distance_km 50-12000; weight_kg 10-9900; volume_m3 0.028-93.213 (historical min-max)
- README follow-ups: business confirms the range rules for "outside the estimation parameters" -> Ruben Cuellar
- features.py signature: get_feature_preprocessor(categorical_columns, numeric_columns), settings per group: none
- Agreed extra parameters: validate_dataframe(allowed_values=None, numeric_ranges=None)
- utils.py helpers: load_csv, save_csv, save_model, load_model
- Environment: no packages beyond pandas, numpy, scikit-learn, joblib, pyyaml (reads config.yaml), wandb (tracking), python-dotenv (reads .env), pytest, pytest-cov, flake8, dvc (data versioning)

## Decisions and owners
- Decisions: quality gate -> mape <= 0.18 on validation; mae reported, not gated (2026-11-09, README release rule)
- Decisions: final model -> trained on the train split only; test split stays sealed (README check, row 2)
- Decisions: out-of-range shipments -> historical min-max as initial rules in validate (README check, row 3)
- Decisions: serving -> batch inference in main.py and real-time API in src/api.py (GET /health, POST /predict), both through clean_dataframe, validate_dataframe and run_inference; Docker image from conda-lock.yml (README section 5)
- Decisions: raw data -> sheet "data" exported once to data/raw/shipments.csv, never committed (README check, row 5)
- Decisions: unknown categories -> validate rejects them first; OneHotEncoder ignores them as a safety net
- Decisions: failed gate -> every run deletes the previous model and predictions first; a failed gate is logged as ERROR and saves nothing (README section 4)
- Decisions: tracking -> main.py owns one W&B run per pipeline run; only a model that passes the gate and writes predictions is logged as artifact shipping_cost_model with alias prod (README section 4)
- Decisions: CI -> ci.yml runs lint, test (90% branch floor), pipeline (dvc pull, stale check, dvc repro) and docker on every pull request into dev or main; W&B registers prod only on pushes to main (README section 6)
- Decisions: release -> a GitHub Release from main runs deploy.yml: release guard, CI on the tag, image with the W&B prod model, smoke test, push to ghcr.io, Render deploy hook, live smoke test; rollback redeploys the image of an older tag (README section 7)
- Decisions: baseline -> no-ML table = median train cost per transport mode, scored on validation in every run (mape 0.560 vs model 0.165 on 2026-11-20); reported, not gated (README section 1)
- Decisions: data versioning -> DVC pointers (*.csv.dvc) and dvc.lock in Git, CSV files and model on DagsHub; dvc repro runs python -m src.main; logs/pipeline.log is not a DVC output (README sections 3 and 4)
- Decisions: new_shipments.csv -> stand-in of the first 20 shipments without cost_usd until the TMS export exists
- Owners (team): utils -> Ruben Cuellar (@rubenc55), branch feature/utils
- Owners (team): load_data -> Ruben Cuellar (@rubenc55), branch feature/load-data
- Owners (team): clean_data -> Ruben Cuellar (@rubenc55), branch feature/clean-data
- Owners (team): validate -> Ruben Cuellar (@rubenc55), branch feature/validate
- Owners (team): features -> Juan (@JELONDONOZ), branch feature/features
- Owners (team): train -> Surya (@jyothikasurya2002), branch feature/train
- Owners (team): evaluate -> Enrique (@puigenrique), branch feature/evaluate
- Owners (team): infer -> Victoria (@victoriasesilva), branch feature/infer
- Owners (team): main -> Ruben Cuellar (@rubenc55), branch feature/main