"""Dataset preparation, feature extraction, and drift batch generation for PhishingShield.

Loads the raw LegitPhish dataset, cleans and validates records, deduplicates URLs
before splitting, creates stratified train/test splits, extracts the standard 16
features using the single shared extractor, outputs processed datasets, and creates
a deliberately shifted synthetic batch for drift monitoring.
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
DEFAULT_CURRENT_BATCH_PATH = "data/processed/current_batch.csv"


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


def make_shifted_batch(
    test_df: pd.DataFrame,
    n: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a deliberately shifted evaluation batch from held-out test data.

    This batch simulates concept and data drift for testing the drift detection
    pipeline (Evidently). It is explicitly SIMULATED and does not represent live
    production traffic.

    Characteristics of the shifted batch:
        - Source: Drawn exclusively from the test split (never training data).
        - Size: Exactly n rows (default 3,000).
        - Class balance: Deliberately shifted to ~80% phishing (target 80% phishing,
          20% legitimate).
        - Feature distribution shift: Introduces sampling bias favoring longer URLs
          and URLs with query parameters via probability weighting.
        - Schema: Preserves identical columns ['url', 'label', <16 FEATURES>].
        - Determinism: Controlled via random_state seed.
        - Non-destructive: Does not mutate the input test DataFrame.

    Args:
        test_df: DataFrame of test split containing ['url', 'label', <FEATURES>].
        n: Total number of rows to sample (default 3000).
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Shifted batch with identical schema to test_df.
    """
    if len(test_df) == 0:
        return test_df.copy()

    # Work on an isolated copy
    df_copy = test_df.copy()

    n_phish = int(round(n * 0.8))
    n_legit = n - n_phish

    phish_pool = df_copy[df_copy["label"] == 1].copy()
    legit_pool = df_copy[df_copy["label"] == 0].copy()

    if len(phish_pool) == 0 or len(legit_pool) == 0:
        raise ValueError("test_df must contain both phishing and legitimate rows.")

    # Calculate sampling weights biasing towards longer URLs and query-heavy URLs
    w_phish = (phish_pool["url_length"].clip(lower=1)) * (1.0 + phish_pool["query_param_count"])
    sum_w_phish = w_phish.sum()
    weights_phish = (w_phish / sum_w_phish) if sum_w_phish > 0 else None

    # Sample phishing rows (without replacement if sufficient pool, else with replacement)
    replace_phish = len(phish_pool) < n_phish
    phish_sampled = phish_pool.sample(
        n=n_phish,
        weights=weights_phish,
        replace=replace_phish,
        random_state=random_state,
    )

    # Sample legitimate rows
    replace_legit = len(legit_pool) < n_legit
    legit_sampled = legit_pool.sample(
        n=n_legit,
        replace=replace_legit,
        random_state=random_state,
    )

    # Combine and shuffle
    shifted_batch = pd.concat([phish_sampled, legit_sampled], axis=0)
    shifted_batch = shifted_batch.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    cols_order = ["url", "label"] + FEATURES
    return shifted_batch[cols_order]


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for data preparation and processing."""
    parser = argparse.ArgumentParser(description="Prepare datasets and extract features.")
    parser.add_argument("--raw-path", default=DEFAULT_RAW_PATH, help="Path to raw CSV file.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH, help="Path to save train CSV.")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to save test CSV.")
    parser.add_argument(
        "--current-batch-path",
        default=DEFAULT_CURRENT_BATCH_PATH,
        help="Path to save simulated drift batch.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion.")
    parser.add_argument("--drift-n", type=int, default=3000, help="Shifted batch size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed.")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise SystemExit(f"Raw dataset file not found at '{args.raw_path}'.")

    # Read raw to compute initial stats
    df_raw = pd.read_csv(raw_path)
    raw_count = len(df_raw)

    url_col = "URL" if "URL" in df_raw.columns else "url"
    label_col = "ClassLabel" if "ClassLabel" in df_raw.columns else "label"

    null_count = int(df_raw[[url_col, label_col]].isna().any(axis=1).sum())
    clean_no_null = df_raw.dropna(subset=[url_col, label_col])
    dupe_count = int(clean_no_null[url_col].astype(str).str.strip().duplicated().sum())

    # Load cleaned and deduplicated dataset
    df_clean = load_dataset(args.raw_path)
    final_count = len(df_clean)

    # Create stratified train and test splits with feature extraction
    train_df, test_df = make_splits(
        df_clean, test_size=args.test_size, random_state=args.random_state
    )

    # Generate simulated drift batch from test split
    current_batch = make_shifted_batch(
        test_df, n=args.drift_n, random_state=args.random_state
    )

    # Ensure output directories exist
    Path(args.train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.current_batch_path).parent.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    train_df.to_csv(args.train_path, index=False)
    test_df.to_csv(args.test_path, index=False)
    current_batch.to_csv(args.current_batch_path, index=False)

    # Compute label distributions
    train_phish = int((train_df["label"] == 1).sum())
    train_legit = int((train_df["label"] == 0).sum())
    test_phish = int((test_df["label"] == 1).sum())
    test_legit = int((test_df["label"] == 0).sum())
    drift_phish = int((current_batch["label"] == 1).sum())
    drift_legit = int((current_batch["label"] == 0).sum())

    print("\n" + "=" * 60)
    print("PHISHING SHIELD - DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Original row count:              {raw_count}")
    print(f"Rows removed due to nulls:       {null_count}")
    print(f"Rows removed due to duplicates:  {dupe_count}")
    print(f"Final clean deduplicated rows:   {final_count}")
    print("-" * 60)
    print(f"Train split rows:                {len(train_df)}")
    print(f"  - Phishing (1):                {train_phish} ({train_phish / len(train_df):.2%})")
    print(f"  - Legitimate (0):              {train_legit} ({train_legit / len(train_df):.2%})")
    print(f"Test split rows:                 {len(test_df)}")
    print(f"  - Phishing (1):                {test_phish} ({test_phish / len(test_df):.2%})")
    print(f"  - Legitimate (0):              {test_legit} ({test_legit / len(test_df):.2%})")
    print("-" * 60)
    print(f"Simulated drift batch rows:      {len(current_batch)}")
    print(f"  - Phishing (1):                {drift_phish} ({drift_phish / len(current_batch):.2%})")
    print(f"  - Legitimate (0):              {drift_legit} ({drift_legit / len(current_batch):.2%})")
    print("-" * 60)
    print(f"Saved train CSV:                 {args.train_path}")
    print(f"Saved test CSV:                  {args.test_path}")
    print(f"Saved drift batch CSV:           {args.current_batch_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
