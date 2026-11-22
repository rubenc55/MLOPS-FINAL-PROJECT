"""Download the model aliased prod from W&B for the release image: python -m scripts.fetch_prod_model

Inputs: SETTINGS["tracking"] (project, model name, prod alias) and WANDB_API_KEY (.env or GitHub Secrets).
Outputs: models/model.joblib (the prod model) and models/model_info.json (its W&B version, run,
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
    except Exception as error:  # report any failure (no key, no prod alias, network) as one clear line
        logger.error("Could not fetch the prod model: %s", error)
        return 1
    logger.info("prod model %s from run %s, commit %s", info["version"], info["wandb_run"], info.get("git_commit"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
