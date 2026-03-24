"""Basic tests to ensure the package is properly installed."""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_cost_scraper_import() -> None:
    """Test that the cost scraper module can be imported."""
    module = importlib.import_module("logs_curator.cost_scraper")
    assert module is not None


def test_package_version_exists() -> None:
    """Test that the package exposes a __version__ attribute."""
    package = importlib.import_module("logs_curator")
    assert hasattr(package, "__version__")
    assert isinstance(package.__version__, str)
