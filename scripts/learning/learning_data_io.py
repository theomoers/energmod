# -*- coding: utf-8 -*-
"""Small helpers for reading learning-data inputs."""

import pandas as pd

from _helpers import create_logger

logger = create_logger(__name__)


def load_historical_capacity(data_file, tech_name, capacity_col="Cumulative production, GW"):
    """Load historical cumulative capacity data from CSV."""
    logger.info("Loading historical data for %s from %s", tech_name, data_file)

    df = pd.read_csv(data_file)

    year_col = next((col for col in df.columns if "year" in col.lower()), None)
    if year_col is None:
        raise ValueError(f"No year column found in {data_file}")

    cap_col = next(
        (
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["cumulative", "capacity", "production"])
        ),
        None,
    )
    if cap_col is None:
        raise ValueError(f"No capacity column found in {data_file}")

    hist_data = df[[year_col, cap_col]].copy()
    hist_data.columns = ["year", "capacity_GW"]
    hist_data = hist_data.dropna()
    hist_data = hist_data.sort_values("year").reset_index(drop=True)

    logger.info(
        "  Loaded %s historical data points (%s-%s)",
        len(hist_data),
        hist_data["year"].min(),
        hist_data["year"].max(),
    )
    return hist_data


def load_historical_cost(data_file, tech_name):
    """Load historical cost-per-unit data from a bundled CSV."""
    logger.info("Loading historical costs for %s from %s", tech_name, data_file)

    df = pd.read_csv(data_file)

    year_col = next((col for col in df.columns if "year" in col.lower()), None)
    if year_col is None:
        raise ValueError(f"No year column found in {data_file}")

    cost_col = next(
        (
            col
            for col in df.columns
            if "cost" in col.lower()
            and "historical" in col.lower()
        ),
        None,
    )
    if cost_col is None:
        cost_col = next(
            (
                col
                for col in df.columns
                if "cost" in col.lower() and "eur" in col.lower()
            ),
            None,
        )
    if cost_col is None:
        raise ValueError(f"No historical cost column found in {data_file}")

    hist_data = df[[year_col, cost_col]].copy()
    hist_data.columns = ["year", "cost_per_unit"]
    hist_data = hist_data.dropna()
    hist_data = hist_data.sort_values("year").reset_index(drop=True)

    logger.info(
        "  Loaded %s historical cost points (%s-%s)",
        len(hist_data),
        hist_data["year"].min(),
        hist_data["year"].max(),
    )
    return hist_data
