import joblib

# Define the paths to the pickle files
model_pipeline_path = "./model/model_pipeline.pkl"

# Load the pickle files
model_pipeline = joblib.load(model_pipeline_path)