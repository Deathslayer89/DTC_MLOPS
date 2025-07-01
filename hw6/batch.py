#!/usr/bin/env python
# coding: utf-8

"""Batch prediction script for NYC taxi ride duration.

The script can be used as a CLI tool (see the ``__main__`` block below) **or**
it can be imported as a regular Python module.  The latter is extremely
helpful for unit and integration testing.

Main entrypoint: ``main``
----------------------------------
``main`` performs the following high-level steps:

1. Resolve input & output paths (supporting both local files and S3).
2. Read the raw parquet data.
3. Prepare the data for the model (see ``prepare_data``).
4. Load a previously-trained scikit-learn model from ``model.bin``.
5. Run predictions and persist the result.

Utility helpers
---------------
* ``get_input_path`` / ``get_output_path`` – make I/O configurable via the
  environment variables ``INPUT_FILE_PATTERN`` and ``OUTPUT_FILE_PATTERN``.
* ``read_data`` / ``save_data`` – thin wrappers around ``pandas`` I/O that
  understand a custom ``S3_ENDPOINT_URL`` (used by Localstack during tests).
* ``prepare_data`` – pure transformation logic that is *very* easy to test.

Keeping the pure transformation isolated from I/O means we can write
fast unit tests without touching the network or the filesystem.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_input_path(year: int, month: int) -> str:
    """Return the input parquet path for the given *year* / *month*.

    The default is the public Taxi Trip data set hosted on CloudFront, but it
    can be overridden via the ``INPUT_FILE_PATTERN`` environment variable.
    """

    default_pattern = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )
    pattern = os.getenv("INPUT_FILE_PATTERN", default_pattern)
    return pattern.format(year=year, month=month)


def get_output_path(year: int, month: int) -> str:
    """Return the output parquet path for the given *year* / *month*.

    The default writes next to the script in an ``output/`` folder, but it can
    be overridden via the ``OUTPUT_FILE_PATTERN`` environment variable.  This
    is helpful for writing to S3 in integration tests.
    """

    default_pattern = (
        "output/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )
    pattern = os.getenv("OUTPUT_FILE_PATTERN", default_pattern)
    return pattern.format(year=year, month=month)


def _build_s3_storage_options() -> Optional[dict]:
    """Return pandas **storage_options** dict when ``S3_ENDPOINT_URL`` is set."""

    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if not endpoint_url:
        return None

    return {"client_kwargs": {"endpoint_url": endpoint_url}}


# ---------------------------------------------------------------------------
# Pure transformation logic
# ---------------------------------------------------------------------------


def prepare_data(df: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
    """Clean and enrich the *df* in-place.

    Steps:
    1. Compute ride duration in minutes.
    2. Filter outliers (1 <= duration <= 60).
    3. Cast *categorical* columns to string type (filling missing with ``-1``).
    """

    df = df.copy()

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_data(path: str, categorical: List[str]) -> pd.DataFrame:
    """Read parquet data from *path* and return a prepared ``DataFrame``."""

    storage_options = _build_s3_storage_options()

    if storage_options:
        df = pd.read_parquet(path, storage_options=storage_options)
    else:
        df = pd.read_parquet(path)

    return prepare_data(df, categorical)


def save_data(df: pd.DataFrame, path: str) -> None:
    """Persist *df* to *path* in parquet format (S3-aware)."""

    storage_options = _build_s3_storage_options()

    if storage_options:
        df.to_parquet(path, engine="pyarrow", index=False, storage_options=storage_options)
    else:
        df.to_parquet(path, engine="pyarrow", index=False)


# ---------------------------------------------------------------------------
# Main batch logic
# ---------------------------------------------------------------------------


def load_model(model_path: str = "model.bin"):
    """Load the *DictVectorizer* and model stored by homework 4."""

    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)

    return dv, model


def main(year: int, month: int) -> None:
    """Run batch prediction for the specified *year* / *month*."""

    categorical: List[str] = ["PULocationID", "DOLocationID"]

    input_path = get_input_path(year, month)
    output_path = get_output_path(year, month)

    # 1. Data  ----------------------------------------------------------------
    df = read_data(input_path, categorical)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    # 2. Prediction -----------------------------------------------------------
    dv, model = load_model()

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    # 3. Persist results -------------------------------------------------------
    df_result = pd.DataFrame({
        "ride_id": df["ride_id"],
        "predicted_duration": y_pred,
    })

    save_data(df_result, output_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python batch.py <year> <month>")

    year_arg = int(sys.argv[1])
    month_arg = int(sys.argv[2])

    main(year_arg, month_arg)