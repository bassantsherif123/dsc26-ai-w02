"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    missing_cols = [col for col in (target, id_col, time_col) if col not in df.columns]
    if missing_cols:
        raise KeyError(missing_cols[0])

    n_rows = int(len(df))

    masks = {
        column: (
            df[column].isna() 
            | df[column].map(
                lambda value: isinstance(value, str) and value.strip() == ""
                )
              )
            for column in df.columns
          }
    
    missing = {
        column: (float(mask.mean()) if n_rows else 0.0) for column, mask in masks.items()
    }

    id_missing = masks[id_col]
    ids = df.loc[~id_missing, id_col]
    duplicate_id_mask = ids.duplicated(keep=False)
    duplicate_ids = sorted(ids.loc[duplicate_id_mask].unique().tolist())

    duplicate_rows = int(df.duplicated(keep="first").sum())

    constant_columns = sorted(
        col for col in df.columns
        if col != target and df.loc[~masks[col], col].nunique(dropna=True) <=1
    )

    missing_targets = masks[target]
    unlabelled_rows = int(missing_targets.sum())
    
    labelled = df.loc[~missing_targets]
    labelled_target = labelled[target]
    n_labelled = len(labelled)
    positive_rate = float((labelled_target == 1).sum() / n_labelled) if n_labelled else 0.0

    time_missing = masks[time_col]
    time_values = df.loc[~time_missing, time_col]
    parsed = pd.to_datetime(time_values, errors="coerce", format="mixed")
    bad_timestamps = int(parsed.isna().sum())
    
    valid_times = parsed.dropna()
    if valid_times.empty:
        time_range = [None, None]
    else:
        time_range = [
            pd.Timestamp(valid_times.min()).isoformat(), 
            pd.Timestamp(valid_times.max()).isoformat()
          ]

    leakage_suspects: list[str] = []

    if n_labelled >= 20 and labelled_target.isin([0, 1]).all() and labelled_target.nunique() == 2:
        y_all = labelled_target.astype(int).to_numpy()
        excluded = {target, id_col, time_col}

        for column in df.columns:
            if column in excluded:
                continue

            series = labelled[column]
            col_missing = masks[column].loc[labelled.index].to_numpy(dtype=bool)
            nonmissing = series.loc[~col_missing]

            # Skip columns with one distinct value, counting missing as a value.
            if nonmissing.nunique() <= 1 and (
                nonmissing.empty or not col_missing.any()
            ):
                continue

            suspect = False

            # (b) Check missingness leakage for every column.
            missing_scores = col_missing.astype(int)

            if np.unique(missing_scores).size > 1:
                auc = float(roc_auc_score(y_all, missing_scores))
                suspect = auc >= 0.98 or auc <= 0.02

            # (a) Check numeric/boolean value leakage.
            if not suspect and (
                pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)
            ):
                if len(nonmissing) >= 20:
                    numeric_values = pd.to_numeric(nonmissing, errors="coerce")
                    y_observed = y_all[~col_missing]
                    values = numeric_values.to_numpy(dtype=float)

                    if (
                        np.unique(y_observed).size == 2
                        and np.isfinite(values).all()
                        and numeric_values.nunique() > 1
                    ):
                        auc = float(roc_auc_score(y_observed, values))
                        suspect = auc >= 0.98 or auc <= 0.02

            if suspect:
                leakage_suspects.append(column)

    leakage_suspects.sort()

    return {
        'n_rows': n_rows,
        'missing': missing,
        'duplicate_ids': duplicate_ids,
        'duplicate_rows': duplicate_rows,
        'constant_columns': constant_columns,
        'unlabelled_rows': unlabelled_rows,
        'positive_rate': positive_rate,
        'bad_timestamps': bad_timestamps,
        'time_range': time_range,
        'leakage_suspects': sorted(leakage_suspects),
    }

    # raise NotImplementedError
