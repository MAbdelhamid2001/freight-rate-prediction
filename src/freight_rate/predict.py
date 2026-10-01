"""Batch inference: produces both deliverables from the saved model.

    uv run freight-predict

1. data/validation.csv  ->  validation_predictions.csv  (load_id,predicted_rate,
   in the template's row order)
2. data/december_chart_inputs.csv  ->  same file, predicted_rate filled in,
   columns and order untouched, as score.py expects.

Both outputs are checked against the same rules score.py enforces before they
are written, so a broken file fails here rather than at submission.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from freight_rate import config
from freight_rate.data import load_csv
from freight_rate.inference import FreightRatePredictor

log = logging.getLogger("freight_rate.predict")

DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def predict_validation(predictor: FreightRatePredictor, validation_path: Path, template_path: Path) -> pd.DataFrame:
    validation = load_csv(validation_path)
    template = pd.read_csv(template_path)

    if validation[config.ID_COL].duplicated().any():
        raise ValueError("validation.csv contains duplicate load_id values")
    if set(template[config.ID_COL]) != set(validation[config.ID_COL]):
        raise ValueError("template and validation.csv do not contain the same load_id values")

    rates = pd.Series(predictor.predict_frame(validation), index=validation[config.ID_COL])
    out = template[[config.ID_COL]].copy()
    out["predicted_rate"] = out[config.ID_COL].map(rates).round(2)
    _check_rates(out["predicted_rate"], "validation predictions")
    return out


def predict_december(predictor: FreightRatePredictor, path: Path) -> pd.DataFrame:
    """Fill the December chart file.

    The file only has pickup, delivery, distance, equipment, weight and date.
    The model also needs coordinates and market features; the predictor fills
    coordinates from its city lookup and the market index from the daily
    December market observed in validation.csv (see features.MarketCalendar).
    """
    december = pd.read_csv(path)
    if list(december.columns) != DECEMBER_COLUMNS:
        raise ValueError(f"{path.name} must have columns {DECEMBER_COLUMNS}")
    inputs = december.drop(columns="predicted_rate")
    december["predicted_rate"] = np.round(predictor.predict_frame(inputs), 2)
    _check_rates(december["predicted_rate"], "December predictions")
    return december[DECEMBER_COLUMNS]


def _check_rates(rates: pd.Series, label: str) -> None:
    if rates.isna().any() or not np.isfinite(rates).all():
        raise ValueError(f"{label}: missing or non-finite predicted_rate")
    if (rates <= 0).any():
        raise ValueError(f"{label}: non-positive predicted_rate")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate validation and December predictions.")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--validation", type=Path, default=config.VALIDATION_PATH)
    parser.add_argument("--template", type=Path, default=config.TEMPLATE_PATH)
    parser.add_argument("--december", type=Path, default=config.DECEMBER_PATH)
    parser.add_argument("--output", type=Path, default=config.PREDICTIONS_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    predictor = FreightRatePredictor.load(args.model)

    predictions = predict_validation(predictor, args.validation, args.template)
    predictions.to_csv(args.output, index=False)
    log.info("wrote %s (%d rows, median $%.0f)", args.output.name, len(predictions),
             predictions["predicted_rate"].median())

    december = predict_december(predictor, args.december)
    december.to_csv(args.december, index=False)
    log.info("filled %s (%d days, $%.0f-$%.0f)", args.december.name, len(december),
             december["predicted_rate"].min(), december["predicted_rate"].max())


if __name__ == "__main__":
    main()
