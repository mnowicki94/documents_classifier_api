import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dataset():
    df = pd.read_csv("data/preprocessed_data.csv", delimiter=";")
    return df


def test_dataset(dataset):
    assert set(dataset["Category"].unique()).issubset({"b", "t", "e", "m"}), (
        "Category column should contain only 'b', 't', 'e', 'm'"
    )


def test_non_empty_columns(dataset):
    for column in dataset.columns:
        if dataset[column].isnull().sum() > 0:
            pytest.warns(UserWarning, f"Column {column} should not have null values")
