"""Tests for src/logger.py: one setup sends every module's logs to the console and the log file."""

import logging

import pytest

from src.logger import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Put the root logger back as it was, so these tests do not change logging for other tests."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_unit_setup_logging_writes_to_file_and_console(tmp_path, capsys):
    """A message from any module logger reaches both the console (stderr) and the log file, in a new folder."""
    log_file = tmp_path / "logs" / "pipeline.log"

    setup_logging(log_file, level="INFO", log_format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("src.train").info("Training model on 10 rows")

    expected = "INFO | src.train | Training model on 10 rows"
    assert expected in log_file.read_text(encoding="utf-8")
    assert expected in capsys.readouterr().err


def test_unit_setup_logging_respects_level_and_does_not_duplicate(tmp_path):
    """Messages below the level are dropped, and calling setup twice keeps one file line per message."""
    log_file = tmp_path / "pipeline.log"

    setup_logging(log_file, level="warning")
    root = setup_logging(log_file, level="warning")
    logging.getLogger("src.main").info("hidden")
    logging.getLogger("src.main").warning("shown")

    content = log_file.read_text(encoding="utf-8")
    assert root.level == logging.WARNING
    assert len(root.handlers) == 2
    assert "hidden" not in content
    assert content.count("shown") == 1


def test_unit_setup_logging_unknown_level_raises(tmp_path):
    """A typo in logging.level stops the run with a clear error and creates no log file."""
    log_file = tmp_path / "logs" / "pipeline.log"

    with pytest.raises(ValueError, match="Unknown log level: 'LOUD'"):
        setup_logging(log_file, level="LOUD")

    assert not log_file.exists()
