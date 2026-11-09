# Team brief

## Business
- Business case: README.md, section 1 (merged into dev on 2026-11-01)
- Use case: Pricing analysts decide the reference cost shared with Sales using the predicted cost_usd
- Notebook: notebooks/MLOps_Cost_Prediction.ipynb
- Target: cost_usd | Problem type: regression | ID column: shipment_id

## Pipeline
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
- Environment: no packages beyond pandas, numpy, scikit-learn, joblib, pytest, flake8 (confirm in item 4)

## Decisions and owners
- Decisions: quality gate -> mape <= 0.18 on validation; mae reported, not gated (2026-11-09, README release rule)
- Decisions: final model -> trained on the train split only; test split stays sealed (README check, row 2)
- Decisions: out-of-range shipments -> historical min-max as initial rules in validate (README check, row 3)
- Decisions: serving -> batch inference now; real-time API in stage 3 (README check, row 4)
- Decisions: raw data -> sheet "data" exported once to data/raw/shipments.csv, never committed (README check, row 5)
- Decisions: unknown categories -> validate rejects them first; OneHotEncoder ignores them as a safety net
- Owners (team): utils -> Ruben Cuellar (@rubenc55), branch feature/utils
- Owners (team): load_data -> Ruben Cuellar (@rubenc55), branch feature/load-data
- Owners (team): clean_data -> Ruben Cuellar (@rubenc55), branch feature/clean-data
- Owners (team): validate -> Ruben Cuellar (@rubenc55), branch feature/validate
- Owners (team): features -> Juan (@JELONDONOZ), branch feature/features
- Owners (team): train -> Surya (@jyothikasurya2002), branch feature/train
- Owners (team): evaluate -> Enrique (@puigenrique), branch feature/evaluate
- Owners (team): infer -> Victoria (@victoriasesilva), branch feature/infer
- Owners (team): main -> Ruben Cuellar (@rubenc55), branch feature/main