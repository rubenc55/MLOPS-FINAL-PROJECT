# Shipping Cost Prediction

> Racing Cargo should fund cost prediction as an MLOps project: tested, weekly retrained estimates replace generic tables, cutting quote time and winning more customers.

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

## 2. Project structure

### The pipeline at a glance

```text
                           main.py  (orchestrator, one command)
                              |
 data/raw/ -> load_data -> clean_data -> validate -> split -> features -> train -> evaluate
                                                (train/val/test) (recipe)   (fit)   (val split)
                                                                              |
                     reports/predictions.csv <- infer <- models/model.joblib (save)
```

### Folders and files

```text
.
├── README.md                 # business case, success metrics, how to run
├── config.yaml               # non-secret settings (paths, target, split sizes)
├── environment.yml           # dependencies (conda)
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets and generated files out of Git
├── LICENSE
├── .github/
│   └── pull_request_template.md
├── notebooks/                # the "lab": exploration only
├── src/                      # the "factory": one job per module
│   ├── load_data.py          # 1. read raw data
│   ├── clean_data.py         # 2. stateless cleaning
│   ├── validate.py           # 3. data quality gate
│   ├── features.py           # 4. unfitted feature recipe
│   ├── train.py              # 5. fit the pipeline
│   ├── evaluate.py           # 6. metrics on unseen data
│   ├── infer.py              # 7. predictions on new data
│   ├── utils.py              # shared read and write helpers
│   └── main.py               # 0. orchestrator: python -m src.main
├── data/  raw/ processed/ inference/
├── models/                   # saved model (generated)
├── reports/                  # predictions, plots (generated)
└── tests/                    # automated tests (stage 2)
```