"""Feature engineering for the Quantium chip data."""
import pandas as pd

BRAND_FIXES = {
    "RED": "RRD",
    "SNBTS": "SUNBITES",
    "INFZNS": "INFUZIONS",
    "WOOLWORTHS": "WW",
    "SMITH": "SMITHS",
    "NCC": "NATURAL",
    "DORITO": "DORITOS",
    "GRNWVES": "GRAIN",
}


def parse_pack_size(names: pd.Series) -> pd.Series:
    """Extract pack size in grams from the product name (e.g. '175g' -> 175.0)."""
    return names.str.extract(r"(\d+)\s*[gG]")[0].astype(float)


def parse_brand(names: pd.Series) -> pd.Series:
    """Return the brand: the first word of the name, uppercased,
    with known spelling variants merged into one standard name."""
    first_word = names.str.split().str[0].str.upper()
    return first_word.replace(BRAND_FIXES)


def add_price_per_pack(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with price_per_pack = TOT_SALES / PROD_QTY."""
    for col in ["TOT_SALES", "PROD_QTY"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    out = df.copy()
    out["price_per_pack"] = out["TOT_SALES"] / out["PROD_QTY"]
    return out