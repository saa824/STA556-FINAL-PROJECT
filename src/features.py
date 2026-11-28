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


def card_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per loyalty card: total sales, total packs, and price per pack.

    Price per pack is total sales divided by total packs, so a card with many
    purchases counts once, not once per purchase.
    """
    required = ["LYLTY_CARD_NBR", "LIFESTAGE", "PREMIUM_CUSTOMER", "TOT_SALES", "PROD_QTY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    cards = (
        df.groupby(["LYLTY_CARD_NBR", "LIFESTAGE", "PREMIUM_CUSTOMER"], as_index=False)
        .agg(total_sales=("TOT_SALES", "sum"), packs=("PROD_QTY", "sum"))
    )
    cards["price_per_pack"] = cards["total_sales"] / cards["packs"]
    return cards