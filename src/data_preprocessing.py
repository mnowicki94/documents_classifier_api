import pandas as pd
import re
import logging
import os
from datetime import datetime
import json

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


def load_dataset(file_path):
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


def check_null_values(df):
    logging.info("Checking for null values")
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            logging.warning(f"Column {column} has null values")


def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespaces
    tokens = text.split()
    return tokens


def preprocess_text_features(df):
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
    return [token for token in url_tokens if token not in title_tokens]


def add_one_hot_encoding(df):
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

    logging.info(f"do_one_hot_encoding: {do_one_hot_encoding}")

    if do_one_hot_encoding:
        # leave only unique tokens from URL
        df["URL"] = df.apply(
            lambda row: get_unique_tokens(row["URL"], row["Title"]), axis=1
        )

        # Add one-hot encoding based on the Publisher and URL_uniques columns
        df = add_one_hot_encoding(df)

    logging.info(f"Final features: {df.columns}")

    # Save the preprocessed dataset to a CSV file
    df.to_csv(output_file, index=False, sep=";")
    logging.info("Data preprocessing completed and saved")
