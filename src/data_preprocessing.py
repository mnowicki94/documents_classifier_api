import pandas as pd
import re
import logging
import os

# Setup logging
log_folder = "/Users/nom3wz/Documents/repos/documents_classifier/logs"
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(log_folder, "data_preprocessing.log"), mode="w"
        ),
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
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return tokens


def preprocess_text_features(df):
    logging.info("Cleaning text features")
    df["URL"] = (
        df["URL"]
        .str.replace("http://", "")
        .str.replace("https://", "")
        .str.replace("www.", "")
        .str.replace(".com", "")
        .str.replace("htm", "")
        .str.replace("html", "")
    )
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
        df[f"URL_{keyword}"] = df["URL_uniques"].apply(
            lambda tokens: int(keyword in tokens)
        )
    return df


def main():
    """
    Main function to preprocess the dataset.

    Steps:
    1. Load the dataset from a CSV file.
    2. Check for null values in the dataset.
    3. Fill null values in the 'Publisher' column with 'unknown'.
    4. Drop unnecessary columns: 'Timestamp', 'Hostname', 'Story', 'ID'.
    5. Preprocess text features in the dataset.
    6. Create a new column 'URL_uniques' with unique tokens from 'URL' and 'Title'.
    7. Drop the 'URL' column.
    8. Add one-hot encoding to categorical features.
    9. Save the preprocessed dataset to a new CSV file.

    Output:
    - A preprocessed CSV file saved to the specified output path.
    """
    file_path = (
        "/Users/nom3wz/Documents/repos/documents_classifier/data/newsCorpora.csv"
    )
    df = load_dataset(file_path)
    check_null_values(df)
    logging.info("Filling null values in 'Publisher' column")
    df["Publisher"] = df["Publisher"].fillna("unknown")
    df = df.drop(["Timestamp", "Hostname", "Story", "ID"], axis=1)
    df = preprocess_text_features(df)
    df["URL_uniques"] = df.apply(
        lambda row: get_unique_tokens(row["URL"], row["Title"]), axis=1
    )
    df = df.drop(["URL"], axis=1)
    df = add_one_hot_encoding(df)
    output_path = (
        "/Users/nom3wz/Documents/repos/documents_classifier/data/data_preprocessed.csv"
    )
    df.to_csv(output_path, index=False, sep=";")
    logging.info("Data preprocessing completed and saved to %s", output_path)
