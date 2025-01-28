import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import joblib
import json

# Setup logging
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"modeling_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_folder, log_filename), mode="w"),
        logging.StreamHandler(),
    ],
)


def transform_text_features(data, text_columns):
    """
    Transforms text features by concatenating specified text columns and applying TF-IDF vectorization.

    Args:
        data (pd.DataFrame): The input dataframe containing text columns.
        text_columns (list of str): List of column names to be concatenated and transformed.

    Returns:
        scipy.sparse.csr_matrix: The transformed sparse matrix with TF-IDF features.
    """
    logging.info("Starting text feature transformation")

    # Concatenate the specified text columns into a single column
    data["combined_text"] = data[text_columns].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )
    logging.info("Text columns concatenated")

    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the combined text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["combined_text"])
    logging.info("TF-IDF vectorization completed")

    # Save the vectorizer
    os.makedirs("models/", exist_ok=True)
    joblib.dump(tfidf_vectorizer, vectorizer_output_file)
    logging.info("Vectorizer saved")

    return tfidf_matrix


def add_other_cols_to_x(extra_cols):
    """
    Adds additional columns to the feature matrix.

    This function takes a list of column names, converts the corresponding columns
    in the dataframe to a sparse matrix, and then combines this sparse matrix with
    the existing TF-IDF matrix.

    Args:
        extra_cols (list of str): List of column names to be added to the feature matrix.

    Returns:
        scipy.sparse.csr_matrix: Combined sparse matrix of the TF-IDF matrix and the additional columns.
    """

    logging.info("Starting to add other columns to the feature matrix")
    logging.info(f"Columns to be added: {extra_cols}")

    # Convert your extra columns to a sparse matrix
    extra_array = df[extra_cols].values
    extra_sparse = csr_matrix(extra_array)
    logging.info("Extra columns converted to sparse matrix")

    # Combine them
    X_combined = hstack([tfidf_matrix, extra_sparse])
    logging.info("Combined TF-IDF matrix with extra columns")

    return X_combined


if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    input_file = config["modeling"]["input_file"]
    model_output_file = config["modeling"]["model_output_file"]
    vectorizer_output_file = config["modeling"]["vectorizer_output_file"]
    test_size = config["modeling"]["test_size"]
    random_state = config["modeling"]["random_state"]
    use_extra_cols = config["modeling"]["use_extra_cols"]
    cols_to_vectorize = config["modeling"]["cols_to_vectorize"]

    # Load the dataset
    df = pd.read_csv(input_file, delimiter=";")
    logging.info("Data loaded successfully from data/data_preprocessed.csv")

    # Transform text features
    logging.info(f"Columns to vectorize: {cols_to_vectorize}")
    tfidf_matrix = transform_text_features(data=df, text_columns=cols_to_vectorize)

    # Check if extra columns need to be added
    logging.info(f"use_extra_cols: {use_extra_cols}")

    if use_extra_cols:
        print("Adding extra columns to the feature matrix")
        X_combined = add_other_cols_to_x(
            [
                "Publisher_tech",
                "URL_tech",
                "Publisher_medical",
                "URL_medical",
                "Publisher_entertainment",
                "URL_entertainment",
                "Publisher_business",
                "URL_business",
            ]
        )
        X = X_combined
    else:
        X = tfidf_matrix

    # Define target
    y = df["Category"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    logging.info("Data split into training and testing sets")
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"test_size: {test_size}, random_state: {random_state}")

    # Initialize and train the model
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    model.fit(X_train, y_train)
    logging.info("Model trained successfully")

    # Save the trained model
    model_filename = os.path.join(model_output_file)
    joblib.dump(model, model_filename)
    logging.info("Model saved")

    # Make predictions and print the classification report
    y_pred = model.predict(X_test)
    logging.info(
        f"Predictions made on the test set {classification_report(y_test, y_pred)}"
    )
