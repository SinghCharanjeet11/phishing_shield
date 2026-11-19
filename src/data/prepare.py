"""Dataset preparation and feature extraction for PhishingShield.

Loads the raw LegitPhish dataset, cleans and validates records, deduplicates URLs
before splitting, creates stratified train/test splits, extracts the standard 16
features using the single shared extractor, and outputs processed datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.extractor import FEATURES, extract_batch

DEFAULT_RAW_PATH = "data/raw/legitphish.csv"
DEFAULT_TRAIN_PATH = "data/processed/train.csv"
DEFAULT_TEST_PATH = "data/processed/test.csv"


def load_dataset(path: str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load and clean the LegitPhish raw dataset.

    Internal label convention:
        1 = Phishing
        0 = Legitimate

    Source LegitPhish convention (verified from Mendeley documentation):
        0.0 = Phishing
        1.0 = Legitimate

    Mappings applied:
        - ClassLabel 0 -> 1 (Phishing)
        - ClassLabel 1 -> 0 (Legitimate)

    Processing steps:
        1. Read raw CSV.
        2. Identify URL and ClassLabel columns.
        3. Drop null or empty URLs.
        4. Strip leading/trailing whitespace from URLs.
        5. Remove records with invalid or missing labels.
        6. Deduplicate URLs before splitting to prevent data leakage.
        7. Return DataFrame with exactly two columns: 'url' (str) and 'label' (int).

    Args:
        path: Filepath to raw CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe with ['url', 'label'] columns.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at '{path}'.")

    df_raw = pd.read_csv(raw_path)

    # Resolve URL and label columns
    url_col = "URL" if "URL" in df_raw.columns else "url"
    label_col = "ClassLabel" if "ClassLabel" in df_raw.columns else "label"

    if url_col not in df_raw.columns:
        raise ValueError(f"URL column not found in {list(df_raw.columns)}")
    if label_col not in df_raw.columns:
        raise ValueError(f"Label column not found in {list(df_raw.columns)}")

    # Keep only relevant columns
    df = df_raw[[url_col, label_col]].copy()
    df.columns = ["url", "raw_label"]

    # Drop null URLs
    df = df.dropna(subset=["url", "raw_label"])

    # Clean URL strings
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]

    # Filter valid labels (0 or 1 in source dataset)
    valid_mask = df["raw_label"].isin([0, 1, 0.0, 1.0])
    df = df[valid_mask].copy()

    # Map labels: source 0 -> internal 1 (phishing), source 1 -> internal 0 (legitimate)
    df["label"] = df["raw_label"].apply(lambda v: 1 if int(v) == 0 else 0).astype(int)
    df = df[["url", "label"]]

    # Deduplicate URLs strictly before splitting to avoid data leakage
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    return df


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train and test splits.

    Ensures no overlapping URLs exist between train and test sets, and preserves
    class distribution.

    Args:
        df: Deduplicated DataFrame with ['url', 'label'] columns.
        test_size: Fraction of dataset allocated to test split (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if len(df) == 0:
        empty_cols = ["url", "label"] + FEATURES
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    train_base, test_base = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"] if df["label"].nunique() > 1 else None,
    )

    train_base = train_base.reset_index(drop=True)
    test_base = test_base.reset_index(drop=True)

    # Extract 16 features using shared extractor
    train_features = extract_batch(train_base["url"])
    test_features = extract_batch(test_base["url"])

    # Assemble complete datasets with exact column order: url, label, <FEATURES>
    cols_order = ["url", "label"] + FEATURES
    train_df = pd.concat([train_base, train_features], axis=1)[cols_order]
    test_df = pd.concat([test_base, test_features], axis=1)[cols_order]

    return train_df, test_df
