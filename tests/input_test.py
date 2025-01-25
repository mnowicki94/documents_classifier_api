import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dataset():
    file_path = (
        "/Users/nom3wz/Documents/repos/documents_classifier/data/newsCorpora.csv"
    )
    df = pd.read_csv(
        file_path,
        delimiter="\t",
        header=None,
        names=[
            "ID",
            "Title",
            "URL",
            "Publisher",
            "Category",
            "Story",
            "Hostname",
            "Timestamp",
        ],
    )
    return df


def test_dataset(dataset):
    assert set(dataset["Category"].unique()).issubset({"b", "t", "e", "m"}), (
        "Category column should contain only 'b', 't', 'e', 'm'"
    )


def test_non_empty_columns(dataset):
    for column in dataset.columns:
        if dataset[column].isnull().sum() > 0:
            pytest.warns(UserWarning, f"Column {column} should not have null values")


def test_unique_ids(dataset):
    assert dataset["ID"].is_unique, "ID column should have unique values"


def test_valid_timestamps(dataset):
    try:
        pd.to_datetime(dataset["Timestamp"])
    except ValueError:
        pytest.fail("Timestamp column should contain valid datetime values")
