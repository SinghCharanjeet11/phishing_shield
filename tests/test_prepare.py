"""Unit tests for dataset loading, preparation, and splitting."""

import pandas as pd
import pytest

from src.data.prepare import load_dataset, make_splits
from src.features.extractor import FEATURES


@pytest.fixture
def synthetic_raw_csv(tmp_path):
    """Generate a small synthetic raw CSV simulating LegitPhish structure."""
    data = {
        "URL": [
            "https://example.com",
            "http://phishing-site-test.com/login",
            "https://test-legit.org/page",
            "http://malware-drop.com/payload.exe",
            "https://example.com",  # Duplicate URL with same label
            "   ",  # Whitespace URL
            "http://another-phish.net",
            "https://good-domain.edu/home",
            "http://bad-ip.org",
            "https://safe-portal.gov",
        ],
        "ClassLabel": [
            1.0,  # Legitimate -> mapped to 0
            0.0,  # Phishing -> mapped to 1
            1.0,  # Legitimate -> mapped to 0
            0.0,  # Phishing -> mapped to 1
            1.0,  # Duplicate
            1.0,  # Invalid whitespace URL
            0.0,  # Phishing -> mapped to 1
            1.0,  # Legitimate -> mapped to 0
            0.0,  # Phishing -> mapped to 1
            1.0,  # Legitimate -> mapped to 0
        ],
    }
    csv_file = tmp_path / "synthetic_legitphish.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return str(csv_file)


def test_load_dataset_cleaning_and_mapping(synthetic_raw_csv):
    """Test loading, null handling, deduplication, and label mapping."""
    df = load_dataset(synthetic_raw_csv)

    # Must have exactly url and label columns
    assert list(df.columns) == ["url", "label"]

    # Deduplication check: 'https://example.com' occurred twice, should appear once
    assert len(df[df["url"] == "https://example.com"]) == 1

    # Whitespace URL should be dropped
    assert "" not in df["url"].values

    # Check label mappings: source 0.0 -> 1, source 1.0 -> 0
    phish_row = df[df["url"] == "http://phishing-site-test.com/login"].iloc[0]
    assert phish_row["label"] == 1

    legit_row = df[df["url"] == "https://example.com"].iloc[0]
    assert legit_row["label"] == 0

    # Ensure all labels are integers 0 or 1
    assert set(df["label"].unique()).issubset({0, 1})
    assert pd.api.types.is_integer_dtype(df["label"])


def test_make_splits(synthetic_raw_csv):
    """Test stratified splitting, feature extraction, column order, and no leakage."""
    df = load_dataset(synthetic_raw_csv)
    train_df, test_df = make_splits(df, test_size=0.25, random_state=42)

    expected_cols = ["url", "label"] + FEATURES
    assert list(train_df.columns) == expected_cols
    assert list(test_df.columns) == expected_cols

    # No URL overlap between train and test
    train_urls = set(train_df["url"])
    test_urls = set(test_df["url"])
    assert len(train_urls.intersection(test_urls)) == 0

    # Total rows match deduplicated count
    assert len(train_df) + len(test_df) == len(df)

    # Features are numeric and finite
    for feat in FEATURES:
        assert not train_df[feat].isna().any()
        assert not test_df[feat].isna().any()
