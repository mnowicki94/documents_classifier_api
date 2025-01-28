"""
Author: Maciej Nowicki
Date: 28.01.2025
Summary: This script preprocesses a dataset by cleaning text features, handling null values, and optionally adding one-hot encoding.
"""

import pandas as pd
import re
import logging
import os
from datetime import datetime
import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import download

# Ensure necessary NLTK resources are downloaded
download("wordnet")
download("omw-1.4")

# Setup logging
os.makedirs("logs/", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"data_preprocessing_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs/", log_filename), mode="w"),
        logging.StreamHandler(),
    ],
)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def load_dataset(file_path):
    """
    Load dataset from a given file path.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logging.info("Loading dataset from %s", file_path)
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


def eda(df):
    """
    Perform exploratory data analysis on the dataset and log key insights.

    Args:
        df (pd.DataFrame): Dataset to analyze.
    """
    logging.info("Starting EDA...")

    # Dataset Count Information
    dataset_count = df["ID"].count()
    logging.info("Dataset Count:\n%s", dataset_count)

    # Check if all observations are unique
    is_unique = df["ID"].is_unique
    logging.info("Check if all observations are unique: %s", is_unique)

    # Summary statistics for Timestamp
    min_timestamp = pd.to_datetime(df["Timestamp"].min(), unit="ms")
    max_timestamp = pd.to_datetime(df["Timestamp"].max(), unit="ms")
    logging.info("Timestamp Summary:\nMin: %s\nMax: %s", min_timestamp, max_timestamp)

    # Missing values
    missing_values = df.isnull().sum()
    logging.info("Missing Values:\n%s", missing_values)

    # Data validation
    assert df["Title"].dtype == object, "Title column should be of type object (text)"
    assert df["Category"].dtype == object, (
        "Category column should be of type object (text)"
    )
    assert set(df["Category"].unique()).issubset({"b", "t", "e", "m"}), (
        "Category column should contain only 'b', 't', 'e', 'm'"
    )
    logging.info("Data validation checks passed.")

    # Distribution of categories with percentages
    category_counts = df["Category"].value_counts()
    total_count = len(df)
    category_percentages = (category_counts / total_count * 100).round(1)

    for category, percentage in category_percentages.items():
        logging.info("Category '%s': %s%%", category, percentage)


def check_null_values(df):
    """
    Check for null values in the dataset and log warnings if any are found.

    Args:
        df (pd.DataFrame): Dataset to check for null values.
    """
    logging.info("Checking for null values")
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            logging.warning(f"Column {column} has null values")


def clean_text(text):
    """
    Clean and lemmatize text.

    Args:
        text (str): Text to clean.

    Returns:
        list: List of lemmatized tokens.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespaces
    tokens = text.split()

    # Apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def preprocess_text_features(df):
    """
    Clean text features in the dataset.

    Args:
        df (pd.DataFrame): Dataset to preprocess.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    logging.info("Cleaning text features")

    # Clean URL column - delete http, https, www, .com, etc
    df["URL"] = (
        df["URL"]
        .str.replace("http://", "")
        .str.replace("https://", "")
        .str.replace("www.", "")
        .str.replace(".com", "")
        .str.replace("htm", "")
        .str.replace("html", "")
    )

    # Clean text features
    text_features = ["Title", "Publisher", "URL"]
    for col in text_features:
        df[col] = df[col].apply(clean_text)
    return df


def get_unique_tokens(url_tokens, title_tokens):
    """
    Get unique tokens from URL that are not in the title.

    Args:
        url_tokens (list): Tokens from the URL.
        title_tokens (list): Tokens from the title.

    Returns:
        list: Unique tokens from the URL.
    """
    return [token for token in url_tokens if token not in title_tokens]


def add_one_hot_encoding(df):
    """
    Add one-hot encoding for specific keywords in the Publisher and URL columns.

    Args:
        df (pd.DataFrame): Dataset to add one-hot encoding to.

    Returns:
        pd.DataFrame: Dataset with one-hot encoding added.
    """
    logging.info("Adding one-hot encoding")
    keywords = ["tech", "medical", "entertainment", "business"]
    for keyword in keywords:
        df[f"Publisher_{keyword}"] = df["Publisher"].apply(
            lambda tokens: int(keyword in tokens)
        )
        df[f"URL_{keyword}"] = df["URL"].apply(lambda tokens: int(keyword in tokens))
    return df


if __name__ == "__main__":
    # Load configuration parameters

    config_path = "config.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    input_path = config["data_preprocessing"]["input_file"]
    output_file = config["data_preprocessing"]["output_file"]
    do_one_hot_encoding = config["data_preprocessing"]["do_one_hot_encoding"]

    # Load dataset
    df = load_dataset(input_path)

    # Perform EDA
    eda(df)

    # Check null values
    check_null_values(df)
    logging.info("Filling null values in 'Publisher' column")

    # Fill null values in Publisher column if null values present
    if df["Publisher"].isnull().sum() > 0:
        df["Publisher"] = df["Publisher"].fillna("unknown")

    # Drop unnecessary columns
    df = df.drop(["Timestamp", "Hostname", "Story", "ID"], axis=1)

    # Preprocess text features
    df = preprocess_text_features(df)

    # Leave only unique tokens in URL column
    for col in ["Title", "Publisher"]:
        df["URL"] = df.apply(
            lambda row: get_unique_tokens(row["URL"], row[col]), axis=1
        )

    logging.info(f"do_one_hot_encoding: {do_one_hot_encoding}")

    if do_one_hot_encoding:
        # Add one-hot encoding based on the Publisher and URL_uniques columns
        df = add_one_hot_encoding(df)

    logging.info(f"Final features: {df.columns}")

    # Save the preprocessed dataset to a CSV file
    df.to_csv(output_file, index=False, sep=";")
    logging.info("Data preprocessing completed and saved")
