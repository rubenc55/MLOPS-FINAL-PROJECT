"""Send every module's log messages to the console and to the pipeline log file.

Inputs: the log file path, level and message format from config.yaml (paths.log_file, logging).
Outputs: the root logger, configured with one console handler and one file handler.
Used by: main (once, at start-up); every other module only calls logging.getLogger(__name__).
Why: the console shows progress while the run is live; logs/pipeline.log keeps an audit trail
of every run (which data, which metrics, whether the gate passed) after the terminal is closed.
"""

import logging
from pathlib import Path


def setup_logging(log_file: Path, level: str = "INFO",
                  log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s") -> logging.Logger:
    """Configure the root logger to write to the console and to log_file (appending).

    Inputs: log_file (e.g. logs/pipeline.log), level name ("DEBUG", "INFO", ...), log_format.
    Outputs: the configured root logger; the log folder is created if missing.
    Why: one call at the entry point; force=True replaces old handlers, so a second run in the same
    process does not write every line twice.
    """
    numeric_level = logging.getLevelName(str(level).upper())
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Unknown log level: {level!r}. Use DEBUG, INFO, WARNING, ERROR or CRITICAL "
            "in config.yaml (logging.level)."
        )
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    return logging.getLogger()
