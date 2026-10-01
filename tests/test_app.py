"""Every Streamlit page renders without raising, on the committed artefacts.

Skipped when models/ or reports/ have not been generated yet (fresh clone
before `freight-train`), because the pages are views over those files.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from freight_rate import config

VIEWS = Path(__file__).resolve().parents[1] / "app" / "views"
ARTEFACTS = (config.MODEL_PATH, config.MODEL_CARD_PATH, config.EDA_SUMMARY_PATH, config.EXPERIMENTS_PATH)

pytestmark = pytest.mark.skipif(not all(p.is_file() for p in ARTEFACTS),
                                reason="trained model / reports not generated yet")


@pytest.mark.parametrize("page", ["overview", "eda", "models", "predict", "december", "batch"])
def test_page_renders(page, monkeypatch):
    monkeypatch.delenv("FREIGHT_API_URL", raising=False)  # embedded mode
    app = AppTest.from_file(str(VIEWS / f"{page}.py"), default_timeout=120)
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.title, f"{page} rendered no title"


def test_predict_page_shows_a_positive_price():
    app = AppTest.from_file(str(VIEWS / "predict.py"), default_timeout=120)
    app.run()
    rate = next(m for m in app.metric if m.label == "Predicted posted rate")
    assert float(rate.value.strip("$").replace(",", "")) > 0


def test_entrypoint_builds_navigation():
    app = AppTest.from_file(str(VIEWS.parent / "streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, [e.message for e in app.exception]
