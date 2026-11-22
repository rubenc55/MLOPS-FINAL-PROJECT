"""Download the model aliased prod from MLflow on DagsHub for the release image: python -m scripts.fetch_prod_model

Inputs: SETTINGS["tracking"] (tracking URI, model name, prod alias) and MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD (.env, or the DagsHub GitHub Secrets in deploy).
Outputs: models/model.joblib (the prod model) and models/model_info.json (its model version, MLflow run,
git commit, data md5 and metrics), at paths.model and paths.model_info in config.yaml.
Used by: deploy.yml, before docker build; by hand to serve the prod model locally.
Why: the release ships the model that passed the gate and was registered as prod, not whatever
model is on the machine that builds the image.
"""

import logging
import sys

from dotenv import load_dotenv

from src.config import PROJECT_ROOT, SETTINGS
from src.logger import setup_logging
from src.tracking import fetch_prod_model

logger = logging.getLogger("fetch_prod_model")


def main() -> int:
    """Download the prod model and its lineage; return 0 on success, 1 with a clear error otherwise."""
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging(SETTINGS["paths"]["log_file"], **SETTINGS["logging"])
    try:
        info = fetch_prod_model(SETTINGS["tracking"], SETTINGS["paths"]["model"], SETTINGS["paths"]["model_info"])
    except Exception as error:  # report any failure (no token, no prod alias, network) as one clear line
        logger.error("Could not fetch the prod model: %s", error)
        return 1
    logger.info("prod model version %s from run %s, commit %s", info["version"], info["run_id"], info["git_commit"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
