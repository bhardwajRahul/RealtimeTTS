import logging
import subprocess
import sys
from pathlib import Path


def test_safepipe_import_does_not_configure_root_logging():
    safepipe_path = (
        Path(__file__).resolve().parents[1]
        / "RealtimeTTS"
        / "engines"
        / "safepipe.py"
    )

    code = """
import importlib.util
import logging
import pathlib
import sys

root = logging.getLogger()
before = (root.level, len(root.handlers))

spec = importlib.util.spec_from_file_location(
    "safepipe_logging_test",
    pathlib.Path(sys.argv[1]),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

after = (root.level, len(root.handlers))
assert before == after == (logging.WARNING, 0), (before, after)
"""

    result = subprocess.run(
        [sys.executable, "-c", code, str(safepipe_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
