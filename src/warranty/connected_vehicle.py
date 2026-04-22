"""Connected-vehicle data ingestion and correlation (Step 9).

If the user has CV data (CAN signals, on-board DTC logs, sensor
readings), they drop files into data/connected/. This module:
  1. Auto-detects CSV or Parquet files and loads them.
  2. Joins to the warranty data by VIN within a configurable time
     window (default 30 days before repair).
  3. Aggregates CV features and feeds them to the LLM to refine
     the root-cause reasoning from Step 8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import failure_log
from .io_utils import ColumnMap, _repo_root, load_settings


def ingest_connected(folder: Optional[Path] = None) -> pd.DataFrame:
    """Load all CSV and Parquet files from the connected-vehicle folder.

    Expects each file to have at least columns: vin, timestamp.
    Long format (vin, timestamp, signal_name, value) is preferred;
    wide format is also accepted and melted automatically.

    Returns an empty DataFrame if the folder is empty or absent.
    """
    folder = folder or (_repo_root() / "data" / "connected")
    if not folder.exists():
        return pd.DataFrame()

    frames = []
    for f in sorted(folder.iterdir()):
        try:
            if f.suffix.lower() == ".csv":
                frames.append(pd.read_csv(f))
            elif f.suffix.lower() == ".parquet":
                frames.append(pd.read_parquet(f))
        except Exception as e:
            failure_log.record(
                step="connected_vehicle.ingest_connected",
                context=f"file={f.name}",
                exc=e,
                lesson="Skipping unreadable CV file; verify format is CSV/Parquet with vin+timestamp.",
            )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalize column names to lowercase.
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure timestamp is datetime.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # If wide format (no signal_name column), melt.
    if "signal_name" not in df.columns and "vin" in df.columns and "timestamp" in df.columns:
        id_vars = ["vin", "timestamp"]
        value_vars = [c for c in df.columns if c not in id_vars]
        if value_vars:
            df = df.melt(id_vars=id_vars, value_vars=value_vars,
                         var_name="signal_name", value_name="value")

    return df


def align_timeseries(
    cv_df: pd.DataFrame,
    warranty_df: pd.DataFrame,
    vin_col: str,
    date_col: str,
    window_days: int | None = None,
) -> pd.DataFrame:
    """Join CV data to warranty claims within a time window.

    For each warranty row (VIN + repair_date), keep CV records from
    [repair_date - window_days, repair_date]. Aggregate numeric signals
    into mean/max/min/std per VIN+repair_date.
    """
    settings = load_settings()
    window_days = window_days or settings.get("connected_vehicle", {}).get("vin_join_window_days", 30)

    if cv_df.empty or "vin" not in cv_df.columns or "timestamp" not in cv_df.columns:
        return pd.DataFrame()

    # Prepare warranty keys.
    wdf = warranty_df[[vin_col, date_col]].copy()
    wdf[date_col] = pd.to_datetime(wdf[date_col], errors="coerce")
    wdf = wdf.rename(columns={vin_col: "vin", date_col: "repair_date"})

    # Merge on VIN.
    merged = cv_df.merge(wdf, on="vin", how="inner")
    # Filter to window.
    merged = merged[
        (merged["timestamp"] >= merged["repair_date"] - pd.Timedelta(days=window_days))
        & (merged["timestamp"] <= merged["repair_date"])
    ]

    if merged.empty:
        return pd.DataFrame()

    # Aggregate by VIN + repair_date + signal.
    if "signal_name" in merged.columns and "value" in merged.columns:
        merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
        agg = (
            merged.groupby(["vin", "repair_date", "signal_name"])["value"]
            .agg(["mean", "max", "min", "std", "count"])
            .reset_index()
        )
        # Pivot to wide: one column per signal × stat.
        agg_wide = agg.pivot_table(
            index=["vin", "repair_date"],
            columns="signal_name",
            values=["mean", "max", "min"],
        )
        agg_wide.columns = [f"cv_{stat}_{sig}" for stat, sig in agg_wide.columns]
        agg_wide = agg_wide.reset_index()
        return agg_wide

    return merged


def correlate_with_warranty(
    warranty_df: pd.DataFrame,
    cv_agg: pd.DataFrame,
    col_map: ColumnMap,
    client: Any | None = None,
) -> pd.DataFrame:
    """Join aggregated CV features back to warranty data.

    If a ClaudeClient is provided, re-evaluate root_cause_reasoning
    with CV context appended.
    """
    if cv_agg.empty:
        return warranty_df

    vin_col = col_map.vin or "vin"
    date_col = col_map.repair_date or "repair_date"

    # Rename to match cv_agg keys.
    wdf = warranty_df.copy()
    merge_left = vin_col
    merge_date = date_col

    # pd.merge_asof needs sorted datetime.
    wdf[merge_date] = pd.to_datetime(wdf[merge_date], errors="coerce")
    cv_agg["repair_date"] = pd.to_datetime(cv_agg["repair_date"], errors="coerce")

    joined = wdf.merge(
        cv_agg,
        left_on=[merge_left, merge_date],
        right_on=["vin", "repair_date"],
        how="left",
        suffixes=("", "_cv"),
    )

    # If client is available, refine root_cause_reasoning.
    if client is not None and "root_cause_reasoning" in joined.columns:
        cv_feature_cols = [c for c in joined.columns if c.startswith("cv_")]
        if cv_feature_cols:
            for idx, row in joined.iterrows():
                cv_summary = "; ".join(
                    f"{c}={row[c]}" for c in cv_feature_cols if pd.notna(row[c])
                )
                if cv_summary:
                    old_reasoning = str(row.get("root_cause_reasoning", ""))
                    try:
                        refined = client.chat(
                            messages=[{
                                "role": "user",
                                "content": (
                                    f"Original root-cause reasoning: {old_reasoning}\n\n"
                                    f"Connected-vehicle data for this VIN (30 days before repair):\n"
                                    f"{cv_summary}\n\n"
                                    "Does the CV data support, contradict, or refine the root cause? "
                                    "Update the reasoning in 2-3 sentences."
                                ),
                            }],
                            model_override=client.heavy_model,
                        )
                        joined.at[idx, "root_cause_reasoning"] = (
                            old_reasoning + " | CV update: " + refined
                        )
                    except Exception as e:
                        failure_log.record(
                            step="connected_vehicle.correlate_with_warranty",
                            context=f"row={idx}",
                            exc=e,
                            lesson="CV-based root-cause refinement failed.",
                        )

    return joined
