"""Model selection and training, tracked end to end in MLflow.

    uv run freight-train                 # full study (~1 h on a 4-core laptop)
    uv run freight-train --quick         # smoke run: fewer trees, 3 Optuna trials
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

The study runs in stages under one parent MLflow run, each candidate as a
nested run, so the MLflow UI can sort every experiment by the same metric:

  1. features   which feature set generalises forward in time (LightGBM)
  2. recency    whether down-weighting old loads helps (sample-weight half-life)
  3. models     the model zoo on the winning features: baseline, spline-ridge,
                random forest, HistGBM, LightGBM, XGBoost
  4. tuning     an Optuna search per family (spline GAM, random forest,
                XGBoost, LightGBM, blend), every trial a nested run
  5. checks     the chosen model under a random split, without the
                label-noise filter, and on unseen cities

    uv run freight-train --extend mlp    # add a model family to an existing study

Every candidate is scored on the same rolling-origin time folds and the winner
is chosen by `config.SELECTION_METRIC`. The winner is refitted on all clean
labelled data, saved to models/, and registered in the MLflow model registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from freight_rate import config, evaluation
from freight_rate.data import load_training_data, load_validation_data, quality_report
from freight_rate.features import FEATURE_SETS, ReferenceData
from freight_rate.models import SAMPLE_WEIGHT_PARAM, build_pipeline, feature_importance
from freight_rate.validation import Fold, evaluate, flag_label_noise, random_folds, time_folds

log = logging.getLogger("freight_rate.train")


# ---------------------------------------------------------------------------
# Candidates and cross-validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    model: str
    feature_set: str = "base"
    half_life_days: float | None = None
    params: dict = field(default_factory=dict)
    filter_noise: bool = True

    @property
    def name(self) -> str:
        hl = f"hl{int(self.half_life_days)}" if self.half_life_days else "hl-none"
        return f"{self.model}|{self.feature_set}|{hl}"

    def mlflow_params(self) -> dict:
        flat = {"model": self.model, "feature_set": self.feature_set,
                "half_life_days": self.half_life_days, "filter_noise": self.filter_noise}
        flat |= {f"hp_{k}": v for k, v in self.params.items()}
        return flat


def recency_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray | None:
    """Exponential decay by age relative to the newest training load."""
    if not half_life_days:
        return None
    age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


def fit_candidate(candidate: Candidate, frame: pd.DataFrame, reference: ReferenceData, quick: bool = False):
    params = dict(candidate.params)
    if quick and candidate.model == "blend":
        params["gbm_n_estimators"] = min(params.get("gbm_n_estimators", 300), 300)
    if quick and candidate.model in {"lightgbm", "xgboost", "hist_gbm", "random_forest"}:
        key = "max_iter" if candidate.model == "hist_gbm" else "n_estimators"
        params[key] = min(params.get(key, 300), 300 if candidate.model != "random_forest" else 60)
    pipeline = build_pipeline(candidate.model, feature_set=candidate.feature_set, reference=reference, params=params)
    weights = recency_weights(frame[config.DATE_COL], candidate.half_life_days)
    fit_kwargs = {SAMPLE_WEIGHT_PARAM: weights} if weights is not None else {}
    pipeline.fit(frame, frame[config.TARGET].to_numpy(dtype=float), **fit_kwargs)
    return pipeline


@dataclass
class CVResult:
    candidate: Candidate
    summary: dict[str, float]
    per_fold: pd.DataFrame
    seconds: float


def cross_validate(candidate: Candidate, frame: pd.DataFrame, reference: ReferenceData,
                   noisy: np.ndarray, folds: list[Fold], quick: bool = False) -> CVResult:
    start = time.perf_counter()
    rows = []
    y = frame[config.TARGET].to_numpy(dtype=float)
    for fold in folds:
        train_idx = fold.train_idx
        if candidate.filter_noise:
            train_idx = train_idx[~noisy[train_idx]]
        pipeline = fit_candidate(candidate, frame.iloc[train_idx], reference, quick=quick)
        pred = pipeline.predict(frame.iloc[fold.test_idx])
        rows.append({"fold": fold.name, "n_train": len(train_idx), "n_test": len(fold.test_idx),
                     **evaluate(y[fold.test_idx], pred, noisy[fold.test_idx])})
    per_fold = pd.DataFrame(rows)
    metric_cols = [c for c in per_fold.columns if c.startswith(("all_", "clean_"))]
    summary = {f"cv_{c}": float(per_fold[c].mean()) for c in metric_cols}
    summary |= {f"cv_{c}_std": float(per_fold[c].std(ddof=0)) for c in ("clean_mape", "all_mae")}
    last = per_fold.iloc[-1]
    summary |= {f"holdout_{c}": float(last[c]) for c in metric_cols}
    return CVResult(candidate, summary, per_fold, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# MLflow plumbing (imported lazily so the runtime package never needs mlflow)
# ---------------------------------------------------------------------------
def _setup_mlflow():
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return mlflow


class Tracker:
    """Collects every CV result and mirrors it to MLflow as a nested run."""

    def __init__(self, mlflow, enabled: bool = True):
        self.mlflow = mlflow
        self.enabled = enabled
        self.results: list[tuple[str, CVResult]] = []

    def record(self, stage: str, result: CVResult, extra_tags: dict | None = None) -> CVResult:
        self.results.append((stage, result))
        log.info("[%s] %-42s cv_clean_mape=%.3f%%  cv_all_mae=%.1f  holdout_clean_mape=%.3f%%  (%.0fs)",
                 stage, result.candidate.name, result.summary["cv_clean_mape"],
                 result.summary["cv_all_mae"], result.summary["holdout_clean_mape"], result.seconds)
        if self.enabled:
            with self.mlflow.start_run(run_name=f"{stage}:{result.candidate.name}", nested=True):
                self.mlflow.set_tags({"stage": stage, **(extra_tags or {})})
                self.mlflow.log_params(result.candidate.mlflow_params())
                self.mlflow.log_metrics(result.summary)
                self.mlflow.log_metric("fit_seconds", result.seconds)
                for step, (_, row) in enumerate(result.per_fold.iterrows()):
                    self.mlflow.log_metrics({"fold_clean_mape": row["clean_mape"],
                                             "fold_all_mae": row["all_mae"]}, step=step)
                self.mlflow.log_text(result.per_fold.to_csv(index=False), "cv_folds.csv")
        return result

    def table(self) -> pd.DataFrame:
        rows = [{"stage": stage, "candidate": r.candidate.name, **r.candidate.mlflow_params(), **r.summary,
                 "fit_seconds": r.seconds, "params_json": json.dumps(r.candidate.params)}
                for stage, r in self.results]
        return pd.DataFrame(rows)


# Families that get their own Optuna search. Each is searched on the winning
# feature set and recency weighting, so the comparison between them is fair.
TUNED_FAMILIES = ("ridge", "random_forest", "xgboost", "lightgbm", "blend", "mlp")


def _best(results: list[CVResult]) -> CVResult:
    return min(results, key=lambda r: r.summary[config.SELECTION_METRIC])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_features(tracker, frame, ref, noisy, folds, quick) -> CVResult:
    """Every feature set under a tree model and an additive model: the two
    families can disagree about which features generalise forward in time."""
    results = [tracker.record("features", cross_validate(Candidate(model, fs), frame, ref, noisy, folds, quick))
               for model in ("lightgbm", "ridge") for fs in FEATURE_SETS]
    return _best(results)


def stage_recency(tracker, best: CVResult, frame, ref, noisy, folds, quick) -> CVResult:
    results = [best]
    for half_life in (30, 60, 120, 240):
        cand = replace(best.candidate, half_life_days=float(half_life))
        results.append(tracker.record("recency", cross_validate(cand, frame, ref, noisy, folds, quick)))
    return _best(results)


def stage_models(tracker, best: CVResult, frame, ref, noisy, folds, quick) -> list[CVResult]:
    """The whole zoo with default hyper-parameters on the winning features."""
    results = [best]
    for model in ("baseline_rpm", "lightgbm", "ridge", "random_forest", "hist_gbm", "xgboost", "blend", "mlp"):
        if model == best.candidate.model:
            continue
        cand = replace(best.candidate, model=model, params={})
        results.append(tracker.record("models", cross_validate(cand, frame, ref, noisy, folds, quick)))
    return results


def search_space(model: str, trial) -> dict:
    """Optuna search space per model family."""
    if model == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True),
            "n_knots": trial.suggest_int("n_knots", 4, 16),
        }
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [12, 18, 24, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if model == "xgboost":
        return {
            "objective": trial.suggest_categorical("objective", ["reg:absoluteerror", "reg:squarederror"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        }
    if model == "lightgbm":
        return {
            "objective": trial.suggest_categorical("objective", ["l1", "huber"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        }
    if model == "mlp":
        return {
            "width": trial.suggest_categorical("width", [64, 128, 256]),
            "depth": trial.suggest_int("depth", 1, 3),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    if model == "blend":
        return {
            "gam_weight": trial.suggest_float("gam_weight", 0.2, 0.9),
            "gam_alpha": trial.suggest_float("gam_alpha", 1e-3, 100, log=True),
            "gam_n_knots": trial.suggest_int("gam_n_knots", 4, 16),
            "gbm_learning_rate": trial.suggest_float("gbm_learning_rate", 0.02, 0.15, log=True),
            "gbm_num_leaves": trial.suggest_int("gbm_num_leaves", 7, 127, log=True),
            "gbm_min_child_samples": trial.suggest_int("gbm_min_child_samples", 5, 200, log=True),
            "gbm_colsample_bytree": trial.suggest_float("gbm_colsample_bytree", 0.4, 1.0),
            "gbm_n_estimators": trial.suggest_int("gbm_n_estimators", 300, 1200, step=100),
        }
    raise ValueError(f"no search space for {model!r}")


def stage_tuning(tracker, model_results: list[CVResult], frame, ref, noisy, folds,
                 n_trials: int, quick: bool) -> list[CVResult]:
    """One Optuna TPE search per family; every trial is a nested MLflow run.

    Returns the best result of each family, the untuned default included, so
    tuning can only improve on it.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    winners = []
    for family in TUNED_FAMILIES:
        defaults = [r for r in model_results if r.candidate.model == family]
        if not defaults:
            continue
        base = defaults[0]
        results: list[CVResult] = [base]

        def objective(trial, base=base, results=results, family=family) -> float:
            cand = replace(base.candidate, params=search_space(family, trial))
            result = tracker.record("tuning", cross_validate(cand, frame, ref, noisy, folds, quick),
                                    extra_tags={"optuna_trial": str(trial.number), "family": family})
            results.append(result)
            return result.summary[config.SELECTION_METRIC]

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials)
        best = _best(results)
        log.info("tuning %s: default %.3f%% -> best %.3f%%", family,
                 base.summary[config.SELECTION_METRIC], best.summary[config.SELECTION_METRIC])
        winners.append(best)
    return winners


def stage_checks(tracker, best: CVResult, frame, ref, noisy, time_fold_list, quick) -> dict:
    """Robustness evidence for the report. Not used for selection."""
    checks: dict[str, float] = {}

    # Random split: same model, shuffled folds. The gap to the time split is
    # the optimism a random split would have hidden.
    rnd = tracker.record("checks", cross_validate(best.candidate, frame, ref, noisy,
                                                  list(random_folds(frame)), quick),
                         extra_tags={"check": "random_split"})
    checks["random_split_cv_clean_mape"] = rnd.summary["cv_clean_mape"]
    checks["time_split_cv_clean_mape"] = best.summary["cv_clean_mape"]

    # No label-noise filter: how much do the corrupted labels hurt training?
    raw = tracker.record("checks", cross_validate(replace(best.candidate, filter_noise=False), frame, ref,
                                                  noisy, time_fold_list, quick),
                         extra_tags={"check": "no_noise_filter"})
    checks["no_noise_filter_cv_clean_mape"] = raw.summary["cv_clean_mape"]

    # Unseen cities: validation.csv has 8 cities never seen in training. Rehearse
    # that on the holdout fold by hiding 8 cities from training entirely.
    fold = time_fold_list[-1]
    rng = np.random.default_rng(config.RANDOM_STATE)
    cities = sorted(set(frame.pickup) | set(frame.delivery))
    hidden = set(rng.choice(cities, size=8, replace=False))
    touches = (frame.pickup.isin(hidden) | frame.delivery.isin(hidden)).to_numpy()
    train_idx = fold.train_idx[~touches[fold.train_idx] & ~noisy[fold.train_idx]]
    pipeline = fit_candidate(best.candidate, frame.iloc[train_idx], ref, quick=quick)
    test = fold.test_idx
    pred = pipeline.predict(frame.iloc[test])
    y = frame[config.TARGET].to_numpy(dtype=float)[test]
    clean = ~noisy[test]
    ape = np.abs(pred - y) / y
    checks["unseen_city_clean_mape"] = float(ape[clean & touches[test]].mean() * 100)
    checks["seen_city_clean_mape"] = float(ape[clean & ~touches[test]].mean() * 100)
    checks["unseen_city_rows"] = int((clean & touches[test]).sum())
    checks["hidden_cities"] = ", ".join(sorted(hidden))
    log.info("checks: %s", checks)
    return checks


# ---------------------------------------------------------------------------
# Final fit, artefacts and registry
# ---------------------------------------------------------------------------
def _data_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()[:16]


def cached_noise_flags(frame: pd.DataFrame, ref: ReferenceData) -> np.ndarray:
    """Out-of-fold noise flags take ~30s; cache them keyed on the data content."""
    config.CACHE_DIR.mkdir(exist_ok=True)
    path = config.CACHE_DIR / f"noise_flags_{_data_fingerprint(frame)}.npy"
    if path.is_file():
        return np.load(path)
    flags = flag_label_noise(frame, ref)
    np.save(path, flags)
    return flags


def build_model_card(best: CVResult, checks: dict, frame: pd.DataFrame, noisy: np.ndarray,
                     ref: ReferenceData, importance: pd.Series | None, quality: dict) -> dict:
    return {
        "model_name": config.REGISTERED_MODEL_NAME,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": best.candidate.name,
        "estimator": best.candidate.model,
        "feature_set": best.candidate.feature_set,
        "features": list(FEATURE_SETS[best.candidate.feature_set]),
        "half_life_days": best.candidate.half_life_days,
        "hyperparameters": best.candidate.params,
        "target": f"log({config.TARGET}), absolute-error objective (median rate)",
        "training_rows": int((~noisy).sum()),
        "dropped_noisy_labels": int(noisy.sum()),
        "training_period": [f"{frame[config.DATE_COL].min():%Y-%m-%d}", f"{frame[config.DATE_COL].max():%Y-%m-%d}"],
        "market_calendar_period": [f"{ref.calendar.first_date:%Y-%m-%d}", f"{ref.calendar.last_date:%Y-%m-%d}"],
        "known_cities": len(ref.cities),
        "validation_design": {
            "scheme": "expanding window, 2-month test horizon",
            "folds": [list(f) for f in config.CV_FOLDS],
            "selection_metric": config.SELECTION_METRIC,
        },
        "cv_metrics": best.summary,
        "cv_per_fold": best.per_fold.to_dict(orient="records"),
        "robustness_checks": checks,
        "feature_importance": importance.round(4).to_dict() if importance is not None else None,
        "data_quality": quality,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }


def finalize(best: CVResult, experiments: pd.DataFrame, tracker: Tracker, mlflow, train: pd.DataFrame,
             validation: pd.DataFrame, reference: ReferenceData, noisy: np.ndarray, folds: list[Fold],
             quality: dict, quick: bool) -> None:
    """Robustness checks, final fit, artefacts, evaluation report and registry for the winner."""
    checks = stage_checks(tracker, best, train, reference, noisy, folds, quick)
    experiments = pd.concat([experiments, tracker.table().iloc[-2:]], ignore_index=True)

    # Final model: every clean labelled load, with the chosen recency weighting.
    final = fit_candidate(best.candidate, train.loc[~noisy], reference, quick=quick)
    importance = feature_importance(final)
    card = build_model_card(best, checks, train, noisy, reference, importance, quality)

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_PATH, compress=3)
    experiments = experiments.sort_values(config.SELECTION_METRIC)
    experiments.to_csv(config.EXPERIMENTS_PATH, index=False)

    # Evaluation & inference report for the chosen model.
    holdout, holdout_frame = evaluation.holdout_evaluation(
        lambda part: fit_candidate(best.candidate, part, reference, quick=quick), train, noisy, folds[-1])
    report = {
        "candidate": best.candidate.name,
        "holdout": holdout,
        "validation_predictions": evaluation.prediction_sanity(final, validation, train, noisy),
        "inference": evaluation.inference_benchmark(final, validation),
    }
    config.EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    card["evaluation"] = {"holdout_clean": holdout["overall_clean"], "inference": report["inference"],
                          "within_5pct_clean": holdout["within_5pct_clean"]}
    config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    evaluation.fig_diagnostics(holdout_frame, f"{holdout['fold']} holdout")
    evaluation.fig_model_comparison(experiments)
    evaluation.fig_feature_importance(importance)
    log.info("holdout %s: clean MAPE %.2f%%, %.1f%% of loads within 5%%; single-load latency %.1f ms",
             holdout["fold"], holdout["overall_clean"]["mape"], holdout["within_5pct_clean"],
             report["inference"]["single_load_ms_p50"])
    log.info("saved %s, %s and %s", config.MODEL_PATH.name, config.MODEL_CARD_PATH.name,
             config.EVALUATION_PATH.name)

    if mlflow:
        mlflow.log_params({f"best_{k}": v for k, v in best.candidate.mlflow_params().items()})
        mlflow.log_metrics({f"best_{k}": v for k, v in best.summary.items()})
        mlflow.log_metrics({k: v for k, v in checks.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"inference_{k}": v for k, v in report["inference"].items()})
        for path in (config.MODEL_CARD_PATH, config.EXPERIMENTS_PATH, config.EVALUATION_PATH):
            mlflow.log_artifact(str(path))
        for name in ("07_holdout_diagnostics", "08_model_comparison", "09_feature_importance"):
            figure = config.FIGURES_DIR / f"{name}.png"
            if figure.is_file():
                mlflow.log_artifact(str(figure), artifact_path="figures")
        _log_and_register(mlflow, final, validation)


def _deployed_result() -> CVResult:
    """The currently deployed model as a CVResult, rebuilt from its model card."""
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    candidate = Candidate(model=card["estimator"], feature_set=card["feature_set"],
                          half_life_days=card.get("half_life_days"), params=card.get("hyperparameters") or {})
    return CVResult(candidate, card["cv_metrics"], pd.DataFrame(card["cv_per_fold"]), 0.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=12, help="Optuna trials per model family (default 12)")
    parser.add_argument("--quick", action="store_true", help="fast smoke run with small models")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument("--extend", nargs="+", metavar="MODEL",
                        help="evaluate and tune only these model families on the deployed model's feature "
                             "set, add them to reports/experiments.csv, and replace the model only if one wins")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.quick:
        args.trials = min(args.trials, 3)

    train = load_training_data()
    validation = load_validation_data()
    # Built from feature columns of both files; the target is never touched.
    reference = ReferenceData.from_frames(train, validation)
    quality = {"train": quality_report(train).as_dict(), "validation": quality_report(validation).as_dict()}

    log.info("flagging corrupted labels out-of-fold ...")
    noisy = cached_noise_flags(train, reference)
    log.info("flagged %d of %d labels (%.2f%%)", noisy.sum(), len(noisy), noisy.mean() * 100)

    folds = list(time_folds(train))
    mlflow = None if args.no_mlflow else _setup_mlflow()
    tracker = Tracker(mlflow, enabled=mlflow is not None)

    from contextlib import nullcontext
    run_name = f"{'extend' if args.extend else 'study'}-{datetime.now():%Y%m%d-%H%M}"
    parent = mlflow.start_run(run_name=run_name) if mlflow else nullcontext()
    with parent:
        if mlflow:
            mlflow.set_tags({"stage": "extend" if args.extend else "study", "quick": str(args.quick)})
            mlflow.log_dict(quality, "data_quality.json")
            mlflow.log_metric("noisy_labels", int(noisy.sum()))

        if args.extend:
            deployed = _deployed_result()
            previous = pd.read_csv(config.EXPERIMENTS_PATH)
            previous = previous[previous.stage != "checks"]
            new_results = [tracker.record("models", cross_validate(
                replace(deployed.candidate, model=model, params={}), train, reference, noisy, folds, args.quick))
                for model in args.extend]
            tuned = stage_tuning(tracker, new_results, train, reference, noisy, folds, args.trials, args.quick)
            challenger = _best(new_results + tuned)
            log.info("challenger %s %.3f%% vs deployed %s %.3f%%", challenger.candidate.name,
                     challenger.summary[config.SELECTION_METRIC], deployed.candidate.name,
                     deployed.summary[config.SELECTION_METRIC])
            experiments = pd.concat([previous, tracker.table()], ignore_index=True)
            if challenger.summary[config.SELECTION_METRIC] < deployed.summary[config.SELECTION_METRIC]:
                log.info("selected %s: it beats the deployed model", challenger.candidate.name)
                finalize(challenger, experiments, tracker, mlflow, train, validation, reference, noisy, folds,
                         quality, args.quick)
            else:
                log.info("kept %s: no challenger beats it", deployed.candidate.name)
                old_checks = pd.read_csv(config.EXPERIMENTS_PATH).query("stage == 'checks'")
                experiments = pd.concat([experiments, old_checks], ignore_index=True)
                experiments.sort_values(config.SELECTION_METRIC).to_csv(config.EXPERIMENTS_PATH, index=False)
                evaluation.fig_model_comparison(experiments)
            return

        best = stage_features(tracker, train, reference, noisy, folds, args.quick)
        best = stage_recency(tracker, best, train, reference, noisy, folds, args.quick)
        model_results = stage_models(tracker, best, train, reference, noisy, folds, args.quick)
        tuned = stage_tuning(tracker, model_results, train, reference, noisy, folds, args.trials, args.quick)
        best = _best(model_results + tuned)
        log.info("selected %s with %s=%.3f%%", best.candidate.name, config.SELECTION_METRIC,
                 best.summary[config.SELECTION_METRIC])
        finalize(best, tracker.table(), tracker, mlflow, train, validation, reference, noisy, folds,
                 quality, args.quick)


def _log_and_register(mlflow, pipeline, sample_frame: pd.DataFrame) -> None:
    """Log the fitted pipeline as an MLflow model and register a new version."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    example = sample_frame[list(config.RAW_FEATURES)].head(5).copy()
    example[config.DATE_COL] = example[config.DATE_COL].dt.strftime("%Y-%m-%d")
    signature = infer_signature(example, pipeline.predict(sample_frame.head(5)))
    package_dir = config.PROJECT_ROOT / "src" / "freight_rate"
    try:
        mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=example,
            code_paths=[str(package_dir)], registered_model_name=config.REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )
    except Exception as exc:  # registry problems must not lose a finished study
        log.warning("MLflow model logging/registration failed: %s", exc)


if __name__ == "__main__":
    main()
