"""Exploratory data analysis: static figures for the report + an aggregate summary.

    uv run freight-eda

Writes
  reports/figures/*.png      figures used in the report and the README
  reports/eda_summary.json   aggregates only (bins, monthly / daily statistics,
                             city coordinates), never raw rows. The Streamlit
                             app renders its EDA page from this file, which is
                             what lets the app be public while the dataset
                             itself stays out of the repository.
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate import config
from freight_rate.data import clean_frame, haversine_miles, load_training_data, load_validation_data, quality_report

log = logging.getLogger("freight_rate.eda")

# Categorical slots 1-3 of the validated reference palette (CVD-safe as a set).
EQUIPMENT_COLORS = {"Dry Van": "#2a78d6", "Reefer": "#eb6834", "Flatbed": "#1baf7a"}
PRIMARY = "#2a78d6"
MUTED = "#8a8984"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CRITICAL = "#e34948"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#b8b7b1", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_target(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(np.log10(train[config.TARGET]), bins=80, color=PRIMARY, edgecolor="#fcfcfb", linewidth=0.4)
    axes[0].set_title("Posted rate is right-skewed: model it on a log scale")
    axes[0].set_xlabel("log10(posted rate $)")
    axes[0].set_ylabel("loads")

    rpm = train[config.TARGET] / train["distance"]
    axes[1].hist(rpm.clip(0, 15), bins=150, color=PRIMARY, edgecolor="none")
    axes[1].set_yscale("log")
    axes[1].axvspan(0, 1, color=CRITICAL, alpha=0.08)
    axes[1].axvspan(5, 15, color=CRITICAL, alpha=0.08)
    axes[1].set_title("Rate per mile: a clean core and two corrupted tails")
    axes[1].set_xlabel("$ per mile (clipped at 15)")
    axes[1].set_ylabel("loads (log scale)")
    _save(fig, "01_target_distribution")


def fig_distance(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sample = train.sample(min(len(train), 12_000), random_state=config.RANDOM_STATE)
    for equipment, color in EQUIPMENT_COLORS.items():
        part = sample[sample.equipment == equipment]
        axes[0].scatter(part.distance, part[config.TARGET], s=3, alpha=0.35, color=color, label=equipment,
                        rasterized=True)
    axes[0].set_yscale("log")
    axes[0].set_ylim(50, 30_000)
    axes[0].set_title("Distance explains most of the rate (corr 0.91)")
    axes[0].set_xlabel("distance (miles)")
    axes[0].set_ylabel("posted rate $ (log)")
    axes[0].legend(markerscale=4, loc="lower right")

    rpm = train[config.TARGET] / train.distance
    bands = pd.cut(train.distance, [0, 250, 500, 750, 1000, 1500, 2000, 3500])
    table = rpm.groupby([bands, train.equipment], observed=True).median().unstack()
    x = np.arange(len(table))
    for equipment, color in EQUIPMENT_COLORS.items():
        axes[1].plot(x, table[equipment], marker="o", markersize=5, color=color, label=equipment)
    axes[1].set_xticks(x, [f"{int(i.left)}-{int(i.right)}" for i in table.index], rotation=30)
    axes[1].set_title("$/mile falls with distance; Reefer > Flatbed > Dry Van")
    axes[1].set_xlabel("distance band (miles)")
    axes[1].set_ylabel("median $ per mile")
    axes[1].legend()
    _save(fig, "02_distance_equipment")


def fig_market(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    both = pd.concat([train.assign(split="train"), validation.assign(split="validation")])
    daily = both.groupby(config.DATE_COL).market_index.mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    axes[0].plot(daily.index, daily.values, color=PRIMARY, linewidth=1.1, label="daily mean")
    axes[0].plot(daily.index, daily.rolling(28, min_periods=1).mean(), color=INK_2, linewidth=2,
                 label="28-day trailing mean")
    axes[0].axvspan(pd.Timestamp("2025-11-01"), daily.index.max(), color=MUTED, alpha=0.12)
    axes[0].text(pd.Timestamp("2025-11-03"), daily.max() * 0.98, "validation\n(Nov-Dec)", color=INK_2, va="top")
    axes[0].set_title("Market index: one value per day with a strong weekly cycle")
    axes[0].set_ylabel("market index")
    axes[0].legend(loc="lower left")

    rpm = train[config.TARGET] / train.distance
    level = rpm.groupby(train[config.DATE_COL]).median()
    axes[1].plot(level.index, level.values, color=PRIMARY, linewidth=1.1)
    axes[1].plot(level.index, level.rolling(28, min_periods=1).mean(), color=INK_2, linewidth=2)
    axes[1].set_title("Daily median $/mile is smooth: rates follow the market trend, not its daily swings")
    axes[1].set_ylabel("median $ per mile")
    _save(fig, "03_market_over_time")


def fig_quality(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    weight = train.weight.dropna()
    axes[0].hist(weight, bins=120, color=PRIMARY, edgecolor="none")
    axes[0].hist(weight[weight < 0], bins=60, color=CRITICAL, edgecolor="none", label="negative (sign error)")
    axes[0].set_title("Weight: negative values, a 47,500 lb cap, 300 missing")
    axes[0].set_xlabel("weight (lb)")
    axes[0].set_ylabel("loads")
    axes[0].legend(loc="upper left")

    crow = haversine_miles(train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon)
    circuity = train.distance / np.maximum(crow, 1)
    axes[1].hist(circuity.clip(1, 3), bins=120, color=PRIMARY, edgecolor="none")
    axes[1].set_yscale("log")
    axes[1].set_title("Road / straight-line distance: a few impossible ratios")
    axes[1].set_xlabel("circuity (clipped at 3)")
    axes[1].set_ylabel("loads (log scale)")
    _save(fig, "04_data_quality")


def fig_cities(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    seen = set(train.pickup) | set(train.delivery)
    coords = _city_table(train, validation)
    fig, ax = plt.subplots(figsize=(9, 5))
    known = coords[coords.city.isin(seen)]
    new = coords[~coords.city.isin(seen)]
    ax.scatter(known.lon, known.lat, s=known.loads / 25, color=PRIMARY, alpha=0.6, label="in training")
    ax.scatter(new.lon, new.lat, s=70, color=CRITICAL, marker="D", label="validation only (unseen)")
    for _, row in new.iterrows():
        ax.annotate(row.city, (row.lon, row.lat), xytext=(5, 4), textcoords="offset points", color=INK, fontsize=8.5)
    ax.set_title("8 validation cities never appear in training: use coordinates, not city IDs")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left")
    _save(fig, "05_cities_seen_unseen")


def fig_signals(train: pd.DataFrame) -> None:
    clean = train[(train[config.TARGET] / train.distance).between(1, 5)]
    rpm = clean[config.TARGET] / clean.distance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.45})
    for equipment, color in EQUIPMENT_COLORS.items():
        part = clean.equipment == equipment
        binned = rpm[part].groupby(pd.qcut(clean.quote_signal[part], 10)).median()
        axes[0].plot([i.mid for i in binned.index], binned.values, marker="o", markersize=5, color=color,
                     label=equipment)
    axes[0].set_title("quote_signal: V-shaped, equipment-specific")
    axes[0].set_xlabel("quote_signal (decile midpoint)")
    axes[0].set_ylabel("median $ per mile")
    axes[0].legend()

    corr = clean.assign(rate_per_mile=rpm)[
        ["rate_per_mile", "distance", "weight", "market_index", "quote_signal"]].corr()
    im = axes[1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right")
    axes[1].set_yticks(range(len(corr)), corr.columns)
    axes[1].grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            axes[1].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(corr.values[i, j]) > 0.6 else INK)
    axes[1].set_title("Linear correlation hides it (0.10)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    _save(fig, "06_signals")


# ---------------------------------------------------------------------------
# Aggregate summary for the app
# ---------------------------------------------------------------------------
def _city_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame in (train, validation):
        for side in ("pickup", "delivery"):
            rows.append(frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    stacked = pd.concat(rows)
    return (stacked.groupby("city").agg(lat=("lat", "first"), lon=("lon", "first"), loads=("lat", "size"))
            .reset_index())


def _hist(values: pd.Series, bins: int, lo: float, hi: float) -> dict:
    counts, edges = np.histogram(values.clip(lo, hi), bins=bins, range=(lo, hi))
    return {"counts": counts.tolist(), "edges": np.round(edges, 4).tolist()}


def build_summary(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    rpm = train[config.TARGET] / train.distance
    seen = set(train.pickup) | set(train.delivery)
    cities = _city_table(train, validation)
    cities["seen_in_training"] = cities.city.isin(seen)

    both = pd.concat([train.assign(split="train"), validation.assign(split="validation")])
    daily = both.groupby(config.DATE_COL).agg(market_index=("market_index", "mean"),
                                              loads=("load_id", "size"),
                                              split=("split", "first"))
    rpm_daily = rpm.groupby(train[config.DATE_COL]).median().rename("median_rpm")
    daily = daily.join(rpm_daily)
    daily.index = daily.index.strftime("%Y-%m-%d")

    monthly = (train.assign(rpm=rpm, month=train[config.DATE_COL].dt.strftime("%Y-%m"))
               .groupby(["month", "equipment"]).agg(median_rpm=("rpm", "median"), loads=("rpm", "size"),
                                                    market_index=("market_index", "mean"))
               .reset_index())

    bands = pd.cut(train.distance, [0, 250, 500, 750, 1000, 1500, 2000, 3500])
    band_table = (rpm.groupby([bands, train.equipment], observed=True)
                  .quantile([0.1, 0.5, 0.9]).unstack().reset_index())
    band_table.columns = ["distance_band", "equipment", "p10", "p50", "p90"]
    band_table["distance_band"] = band_table.distance_band.astype(str)

    return {
        "rows": {"train": len(train), "validation": len(validation)},
        "periods": {
            "train": [f"{train.date.min():%Y-%m-%d}", f"{train.date.max():%Y-%m-%d}"],
            "validation": [f"{validation.date.min():%Y-%m-%d}", f"{validation.date.max():%Y-%m-%d}"],
        },
        "equipment_share": {
            "train": train.equipment.value_counts(normalize=True).round(4).to_dict(),
            "validation": validation.equipment.value_counts(normalize=True).round(4).to_dict(),
        },
        "quality": {"train": quality_report(train).as_dict(), "validation": quality_report(validation).as_dict()},
        "target": {
            "describe": train[config.TARGET].describe().round(2).to_dict(),
            "log10_hist": _hist(np.log10(train[config.TARGET]), 60, 1.7, 4.5),
            "rpm_hist": _hist(rpm, 120, 0, 15),
            "rpm_quantiles": rpm.quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]).round(3).to_dict(),
        },
        "weight_hist": _hist(train.weight.dropna(), 100, -47_500, 47_500),
        "market_shift": {
            "train_mean": round(float(train.market_index.mean()), 4),
            "validation_mean": round(float(validation.market_index.mean()), 4),
            "within_day_std": round(float(train.groupby("date").market_index.std().mean()), 4),
            "across_day_std": round(float(train.groupby("date").market_index.mean().std()), 4),
        },
        "correlations": train.assign(rate_per_mile=rpm)[
            [config.TARGET, "rate_per_mile", "distance", "weight", "market_index", "quote_signal"]]
            .corr().round(3).to_dict(),
        "daily": daily.reset_index(names="date").to_dict(orient="records"),
        "monthly": monthly.round(4).to_dict(orient="records"),
        "distance_bands": band_table.round(4).to_dict(orient="records"),
        "cities": cities.round(5).to_dict(orient="records"),
        "unseen_city_rows_in_validation": int(
            (~validation.pickup.isin(seen) | ~validation.delivery.isin(seen)).sum()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    _style()
    train = load_training_data()
    validation = load_validation_data()

    fig_target(train)
    fig_distance(train)
    fig_market(train, validation)
    fig_quality(train)
    fig_cities(train, validation)
    fig_signals(clean_frame(train))

    summary = build_summary(train, validation)
    config.EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log.info("wrote %d figures to %s and %s", len(list(config.FIGURES_DIR.glob("*.png"))),
             config.FIGURES_DIR, config.EDA_SUMMARY_PATH.name)


if __name__ == "__main__":
    main()
