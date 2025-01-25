import logging
from ing_mle_task_model.preprocessing import DataPreprocessor
from ing_mle_task_model.training import BuildModel
from ing_mle_task_model.config.logconfig import setup_logging

# Configure logging
setup_logging(file_log_level=logging.DEBUG,
                console_log_level=logging.INFO)

# 1. Load, analyse and preprocess input data for model training
preprocessor = DataPreprocessor('./input/newsCorpora.csv')
preprocessor.load_data(headers=["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"])
preprocessor.load_missing_values_dict("./ing_mle_task_model/missing_values_handle.json")
preprocessor.mark_as_missing()
preprocessor.check_missing_values(drop_missing=True)
preprocessor.check_columns(columns=["title", "category"])
preprocessor.drop_columns(columns=["id", "url", "publisher", "story", "hostname", "timestamp"])
preprocessor.check_duplicates(columns=["title", "category"], drop_duplicates=True)
preprocessor.check_imbalance(column="category")
preprocessed_data = preprocessor.get_dataframe()


# 2. Train, evaluate the model and if acceptable, save it as pickle
reglog_model = BuildModel(df=preprocessed_data)
reglog_model.split_data(test_size=0.2)

param_grid = {
    'tfidf__tokenizer': [None],
    'tfidf__stop_words': ['english'],
    'tfidf__max_features': [10000, 20000],
    'tfidf__lowercase': [True],
    'clf__multi_class': ['multinomial'], 
    'clf__max_iter': [1000, 2000],
    'clf__class_weight': [None, 'balanced'],
    'clf__solver': ['lbfgs', 'newton-cg'],
    'clf__C': [1]
}

reglog_model.train_model(param_grid, scoring='f1_macro', cv_splits=5)
reglog_model.evaluate_model(run_cross_validation=True)
reglog_model.create_pickle(path="./output/model_pipeline.pkl")

