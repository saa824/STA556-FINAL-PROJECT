"""Cleaning functions for the Quantium chip data."""
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if expected columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def standardize_product_names(names: pd.Series) -> pd.Series:
    """Collapse repeated whitespace and trim product names."""
    return names.str.replace(r"\s+", " ", regex=True).str.strip()


def remove_non_chip_products(
    df: pd.DataFrame, pattern: str = "salsa"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop products whose name matches pattern. Returns (kept, removed)."""
    _require_columns(df, ["PROD_NAME"])
    is_non_chip = df["PROD_NAME"].str.contains(pattern, case=False)
    return df.loc[~is_non_chip].copy(), df.loc[is_non_chip].copy()


def remove_outliers(
    df: pd.DataFrame, qty_col: str = "PROD_QTY", max_qty: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop lines with quantity above max_qty. Returns (kept, removed)."""
    _require_columns(df, [qty_col])
    is_outlier = df[qty_col] > max_qty
    return df.loc[~is_outlier].copy(), df.loc[is_outlier].copy()