import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure logging to log to a file in the log folder
import os

# Setup logging
log_folder = "/Users/nom3wz/Documents/repos/documents_classifier/logs"
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_folder, "modeling.log"), mode="w"),
        logging.StreamHandler(),
    ],
)

# Load the dataset
df = pd.read_csv("data/data_preprocessed.csv", delimiter=";")
logging.info("Data loaded successfully from data/data_preprocessed.csv")


def transform_text_features(data):
    """
    Transforms text features by concatenating text columns and applying TF-IDF vectorization.

    Args:
        data (pd.DataFrame): The input dataframe containing text columns.

    Returns:
        pd.DataFrame: The transformed dataframe with TF-IDF features.
    """
    logging.info("Starting text feature transformation")

    # Concatenate the text columns into a single column
    data["combined_text"] = data["Title"] + " " + data["URL_uniques"]
    logging.info("Text columns concatenated")

    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the combined text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["combined_text"])
    logging.info("TF-IDF vectorization completed")

    return tfidf_matrix


# Transform text features
tfidf_matrix = transform_text_features(data=df)

# Define features and target
X = tfidf_matrix
y = df["Category"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info("Data split into training and testing sets")

# Initialize and train the model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
model.fit(X_train, y_train)
logging.info("Model trained successfully")

# Make predictions and print the classification report
y_pred = model.predict(X_test)
logging.info("Predictions made on the test set")
print(classification_report(y_test, y_pred))
