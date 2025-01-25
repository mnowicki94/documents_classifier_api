import pytest
import pandas as pd
import numpy as np
from io import StringIO
import json
from ing_mle_task_model.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    return """1\tTitle1\turl1\tPublisher1\tCategory1\tStory1\tHostname1\tTimestamp1
2\tTitle2\turl2\tPublisher2\tCategory2\tStory2\tHostname2\tTimestamp2"""

@pytest.fixture
def preprocessor():
    return DataPreprocessor(file_path="dummy_path")

def test_check_missing_values(sample_data, preprocessor):
    headers = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]
    preprocessor.dataframe = pd.read_csv(StringIO(sample_data), delimiter='\t', header=None, names=headers)
    preprocessor.missing_values_dict = {"title": ["Title1"], "publisher": ["Publisher2"]}
    preprocessor.mark_as_missing()
    preprocessor.check_missing_values()
    missing_values = preprocessor.dataframe.isnull().sum()
    assert missing_values['title'] == 1
    assert missing_values['publisher'] == 1

def test_check_imbalance(sample_data, preprocessor):
    headers = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]
    preprocessor.dataframe = pd.read_csv(StringIO(sample_data), delimiter='\t', header=None, names=headers)
    preprocessor.check_imbalance("publisher", imbalance_threshold=0.5)
    value_counts = preprocessor.dataframe["publisher"].value_counts(normalize=True)
    assert value_counts["Publisher1"] == 0.5
    assert value_counts["Publisher2"] == 0.5

def test_check_duplicates(sample_data, preprocessor):
    headers = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]
    preprocessor.dataframe = pd.read_csv(StringIO(sample_data), delimiter='\t', header=None, names=headers)
    preprocessor.dataframe = pd.concat([preprocessor.dataframe, preprocessor.dataframe.iloc[[0]]], ignore_index=True)
    initial_length = len(preprocessor.dataframe)
    preprocessor.check_duplicates(["id", "title"])
    final_length = len(preprocessor.dataframe)
    assert final_length == initial_length - 1