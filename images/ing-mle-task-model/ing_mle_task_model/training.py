from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import logging

logger = logging.getLogger(__name__)

class BuildModel:
    def __init__(self, df):
        """
        Initializes the BuildModel class with a DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data for training the model. 
                               It should have columns 'title' for features and 'category' for labels.

        Attributes:
            model_pipe (Pipeline): A scikit-learn pipeline with TfidfVectorizer and LogisticRegression.
            X (pd.Series): The feature data extracted from the 'title' column of the DataFrame.
            y (pd.Series): The target labels extracted from the 'category' column of the DataFrame.
            X_train (pd.Series or None): Placeholder for the training feature data.
            X_test (pd.Series or None): Placeholder for the testing feature data.
            y_train (pd.Series or None): Placeholder for the training target labels.
            y_test (pd.Series or None): Placeholder for the testing target labels.
            y_pred (pd.Series or None): Placeholder for the predicted labels.
            best_model (Pipeline or None): Placeholder for the best model after training.
            accuracy (float or None): Placeholder for the accuracy of the model.
        """
        self.model_pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
        self.X = df['title']
        self.y = df['category']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.best_model = None
        self.accuracy = None
        logger.info(f"BuildModel object created.")

    def split_data(self, test_size=0.2):
        """
        Splits the dataset into training and testing sets.

        Parameters:
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.

        Returns:
        None: The method updates the instance variables X_train, X_test, y_train, and y_test with the split data.

        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        logger.info(f"Train-test split performed with test size: {test_size}")
    
    def train_model(self, param_grid: dict, cv_splits=5, scoring='f1_macro'):
        """
        Trains the model using GridSearchCV with the provided parameter grid and cross-validation settings.

        Args:
            param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            cv_splits (int, optional): Number of splits for cross-validation. Default is 5.
            scoring (str, optional): Scoring metric to use for evaluating the model. Default is 'f1_macro'.

        Raises:
            Exception: If an error occurs during the grid search process.
        """
        stratified_kfold = StratifiedKFold(n_splits=cv_splits)
        grid_search = GridSearchCV(self.model_pipe, param_grid, cv=stratified_kfold, scoring=scoring, n_jobs=-1, verbose=3)
        logger.info("Grid Search started.")
        try:
            grid_search.fit(self.X_train, self.y_train)
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Best Score: {grid_search.best_score_}")
            self.best_model = grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during grid search: {e}")
            raise

    def evaluate_model(self, threshold_acc=0.85, run_cross_validation=True):
        """
        Evaluate the performance of the trained model on the test dataset.

        Parameters:
        threshold_acc (float): The accuracy threshold to determine if the model's performance is acceptable. Default is 0.85.
        run_cross_validation (bool): Flag to indicate whether to run cross-validation on the test dataset. Default is True.

        """
        self.y_pred = self.best_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        logger.info(f"Accuracy: {self.accuracy}")
        logger.info(f"Classification Report:\n{report}")
        if self.accuracy >= threshold_acc:
            logger.info(f"Model performance is acceptable (accuracy higher than {threshold_acc})")
        if run_cross_validation:
            logger.info("Running cross-validation for best model on test dataset.")
            self.cross_validate_on_test(cv=5)
        logger.info("Model evaluation complete.")

    def cross_validate_on_test(self, cv=5):
        """
        Perform cross-validation on the test dataset using the best model.

        Parameters:
        cv (int): Number of cross-validation folds. Default is 5.

        Returns:
        None

        Logs:
        - Cross-validation scores for each fold.
        - Mean cross-validation score.
        - Standard deviation of cross-validation scores.
        - Completion message for cross-validation on the best model.
        """
        stratified_kfold = StratifiedKFold(n_splits=cv)
        scores = cross_val_score(self.best_model, self.X_test, self.y_test, scoring='f1_macro', cv=stratified_kfold, n_jobs=-1, verbose=3)
        logger.info(f"Cross-validation scores on test dataset: {scores}")
        logger.info(f"Mean cross-validation score: {scores.mean()}")
        logger.info(f"Standard deviation of cross-validation scores: {scores.std()}")
        logger.info(f"Cross-validation on best model (selected from grid search) completed.")

    def create_pickle(self, path="./output/model_pipeline.pkl"):
        """
        Save the trained model to a pickle file.

        Parameters:
        path (str): The file path where the model will be saved. Default is "./output/model.pkl".

        Raises:
        ValueError: If the model has not been evaluated or if the model accuracy is below 0.8.

        """
        if self.accuracy is None:
            logger.error("Model must be evaluated before saving.")
            raise ValueError("Model must be evaluated before saving.")
        if self.accuracy < 0.8:
            logger.error("Model accuracy is too low to save.")
            raise ValueError("Model accuracy is too low to save.")
        with open(path, 'wb') as file:
            pickle.dump(self.best_model, file)
        logger.info(f"Model saved to {path}")