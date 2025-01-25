import pandas as pd
import numpy as np
import logging
import json


logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initializes the DataPreprocessor object.

        Args:
            file_path (str): The path to the file that will be processed.

        Attributes:
            file_path (str): Stores the path to the file.
            dataframe (pd.DataFrame or None): Placeholder for the dataframe that will be created from the file.
        """
        self.file_path = file_path
        self.dataframe = None
        logger.info(f"DataPreprocessor object created with file path: {file_path}")

    def load_data(self, headers: list):
        """
        Loads data from a CSV file into a pandas DataFrame.

        Args:
            headers (list): A list of column names for the DataFrame.

        Raises:
            Exception: If there is an error loading the CSV file.

        """
        try:
            headers = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]            
            self.dataframe = pd.read_csv(self.file_path, delimiter='\t', header=None, names=headers)
            logger.info(f"CSV file loaded successfully from {self.file_path}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def load_missing_values_dict(self, json_file_path: str):
        """
        Load a dictionary of missing values from a JSON file.

        This method reads a JSON file from the specified path and loads its content
        into the `missing_values_dict` attribute of the class instance. If the file
        cannot be read or parsed, an error is logged and the exception is raised.

        Args:
            json_file_path (str): The file path to the JSON file containing the missing values dictionary.

        Raises:
            Exception: If there is an error reading or parsing the JSON file.
        """
        try:
            with open(json_file_path, 'r') as file:
                self.missing_values_dict = json.load(file)
            logger.info("Missing values dictionary loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading missing values dictionary: {e}")
            raise

    def mark_as_missing(self):
        """
        Replaces specified missing values in the dataframe with NaN.

        This method iterates over the columns of the dataframe and replaces the 
        values specified in the `missing_values_dict` with NaN. If the dataframe 
        is not loaded, it logs an error message.

        Raises:
            Exception: If an error occurs during the replacement process, it logs 
            the error and re-raises the exception.

        """
        try:
            if self.dataframe is not None:
                for column, missing_values in self.missing_values_dict.items():
                    if column in self.dataframe.columns:
                        self.dataframe[column] = self.dataframe[column].replace(missing_values, np.nan)
                logger.info("Specified missing values replaced with NaN.")
            else:
                logger.error("Dataframe is not loaded. Please load the data first.")
        except Exception as e:
            logger.error(f"Error replacing missing values: {e}")
            raise

    def check_missing_values(self, drop_missing=False):
        """
        Check for missing values in the dataframe and optionally drop rows with missing values.
        This method logs the number of missing values in each column of the dataframe. If there are rows with missing values,
        it logs those rows. If the `drop_missing` parameter is set to True, it drops the rows with missing values from the dataframe
        and logs the shape of the dataframe before and after dropping the rows.
        Parameters:
        -----------
        drop_missing : bool, optional
            If True, rows with missing values will be dropped from the dataframe. Default is False.
        Raises:
        -------
        RuntimeError:
            If the dataframe is not loaded.
        Exception:
            If any other error occurs during the execution of the method.
        """
        try:
            if self.dataframe is not None:
                missing_values = self.dataframe.isnull().sum()
                logger.info("Missing values in each column:")
                logger.info(missing_values)
                
                # Log rows with missing values
                rows_with_missing = self.dataframe[self.dataframe.isnull().any(axis=1)]
                if not rows_with_missing.empty:
                    logger.info("Rows with missing values:")
                    for index, row in rows_with_missing.iterrows():
                        logger.debug(f"Row {index}: {row.to_dict()}")
                    
                    if drop_missing:
                        initial_shape = self.dataframe.shape
                        self.dataframe.dropna(inplace=True)
                        logger.info(f"Dropped rows with missing values. Shape before: {initial_shape}, after: {self.dataframe.shape}")
                else:
                    logger.info("No rows with missing values found.")
            else:
                logger.error("Dataframe is not loaded. Please load the data first.")
                raise RuntimeError("Dataframe is not loaded. Please load the data first.")
        except Exception as e:
            logger.error(f"Error checking missing values: {e}")
            raise

    def check_columns(self, columns: str):
        """
        Check and log the unique value counts for each column in the provided list of columns.

        Args:
            columns (str): A string representing the column names to check.

        Raises:
            RuntimeError: If the dataframe is not loaded.
            Exception: If there is an error while checking unique values.
        """
        for col in columns:
            try:
                if self.dataframe is not None:
                    col_counts = self.dataframe[col].value_counts()
                    logger.info(f"Unique values count in {col} column:")
                    logger.info(col_counts)
                else:
                    logger.error("Dataframe is not loaded. Please load the data first.")
                    raise RuntimeError("Dataframe is not loaded. Please load the data first.")
            except Exception as e:
                logger.error(f"Error checking unique values: {e}")
                raise

    def drop_columns(self, columns: list):
        """
        Drops specified columns from the dataframe.

        Parameters:
        columns (list): List of column names to be dropped from the dataframe.

        Raises:
        Exception: If there is an error while dropping the columns.
        RuntimeError: If the dataframe is not loaded.

        """
        try:
            if self.dataframe is not None:
                self.dataframe = self.dataframe.drop(columns, axis=1)
                logger.info(f"Columns {columns} dropped successfully.")
            else:
                logger.error("Dataframe is not loaded. Please load the data first.")
                raise RuntimeError("Dataframe is not loaded. Please load the data first.")
        except Exception as e:
            logger.error(f"Error dropping columns: {e}")
            raise

    def check_imbalance(self, column: str, imbalance_threshold=0.20):
        """
        Check for imbalance in a specified column of the dataframe.
        This method calculates the distribution of values in the specified column and logs the distribution.
        It then checks if any value's proportion is below the given imbalance threshold and logs a warning if so.
        Args:
            column (str): The name of the column to check for imbalance.
            imbalance_threshold (float, optional): The threshold below which a value's proportion is considered imbalanced. Defaults to 0.20.
        Raises:
            ValueError: If the specified column does not exist in the dataframe.
            RuntimeError: If the dataframe is not loaded.
        """
        try:
            if self.dataframe is not None:
                if column in self.dataframe.columns:
                    value_counts = self.dataframe[column].value_counts(normalize=True)
                    logger.info(f"Distribution of values in {column} column:")
                    logger.info(value_counts)
                    
                    # Check for imbalance
                    imbalanced_values = value_counts[value_counts < imbalance_threshold]
                    if not imbalanced_values.empty:
                        for value, proportion in imbalanced_values.items():
                            logger.warning(f"Data is imbalanced in {column} column for value '{value}' with proportion {proportion} - because lower the threshold {imbalance_threshold}")
                    else:
                        logger.info(f"Data is balanced in {column} column.")
                else:
                    logger.error(f"Column {column} does not exist in the dataframe.")
                    raise ValueError(f"Column {column} does not exist in the dataframe.")
            else:
                logger.error("Dataframe is not loaded. Please load the data first.")
                raise RuntimeError("Dataframe is not loaded. Please load the data first.")
        except Exception as e:
            logger.error(f"Error checking imbalance: {e}")

    def check_duplicates(self, columns: list, drop_duplicates=True):
        """
        Check for duplicate rows in the dataframe based on specified columns and optionally drop them.
        Parameters:
        columns (list): List of column names to check for duplicates.
        drop_duplicates (bool): If True, drop duplicate rows based on the specified columns. Default is True.
        Raises:
        RuntimeError: If the dataframe is not loaded.
        Exception: If an error occurs during the duplicate check process.
        """
        try:
            if self.dataframe is not None:
                if all(col in self.dataframe.columns for col in columns):
                    duplicate_rows = self.dataframe[self.dataframe.duplicated(subset=columns)]
                    num_duplicates = duplicate_rows.shape[0]
                    logger.info(f"Number of duplicate rows based on {columns}: {num_duplicates}")
                    
                    if num_duplicates > 0:
                        logger.debug("Duplicate rows:")
                        logger.debug(duplicate_rows)
                        
                        if drop_duplicates:
                            initial_shape = self.dataframe.shape
                            self.dataframe.drop_duplicates(subset=columns, inplace=True)
                            logger.info(f"Dropped duplicate rows. Shape before: {initial_shape}, after: {self.dataframe.shape}")
                    else:
                        logger.info("No duplicate rows found based on the specified column.")
                else:
                    logger.error(f"Columns {columns} do not exist in the dataframe.")
            else:
                logger.error("Dataframe is not loaded. Please load the data first.")
                raise RuntimeError("Dataframe is not loaded. Please load the data first.")
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            raise

    def get_dataframe(self):
        """
        Retrieves the dataframe if it has been loaded.

        Returns:
            pd.DataFrame: The loaded dataframe.

        Raises:
            RuntimeError: If the dataframe is not loaded.
        """
        if self.dataframe is None:
            logger.error("Dataframe is not loaded. Please load the data first.")
            raise RuntimeError("Dataframe is not loaded. Please load the data first.")
        return self.dataframe