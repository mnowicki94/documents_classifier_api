# Machine Learning Engineer Task - Document Classifier 

## Background
We receive several million documents a year that need to be classified and distributed to the correct department. To speed up this process, we want to develop a machine learning model to classify documents automatically. Since we cannot share real data for this task, we use a substitute data set consisting of news headlines.

## Goal
1. Build a predictive model to classify headlines into 4 categories (business, science and technology, entertainment, and health).
2. Create a REST API to serve the model.
3. Write an example script (Bash or Python) that uses the API from the command line.
4. Prepare a presentation of your approach and results.

## Data
The news headlines data set can be found at: [News Aggregator Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip)

This dataset consists of the following columns, delimited by `\t`:
- `id`: Numeric ID
- `title`: News title (to be classified)
- `url`: URL of the original article
- `publisher`: Name of the publisher
- `category`: Category (b for business, t for science and technology, e for entertainment, and m for health)
- `story`: Alphanumeric id of the cluster that includes news about the same story
- `hostname`: Hostname
- `timestamp`: Timestamp



# Setting Up the Environment

To create the Conda environment and install the required packages, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mnowicki94/documents_classifier_api.git
   cd https://github.com/mnowicki94/documents_classifier_api.git

2. **Create the Conda Environment**:

    conda env create -f environment.yml

3. **Activate the Environment**:

    conda activate ml_doc_classifier_env

You should now have a reproducible environment set up and ready to use.

By following these steps, you ensure that anyone who clones your repository can easily set up the same environment and work with your project.


# Running whole pipeline

1. ** in terminal:
    chmod +x run_pipeline.sh
    ./run_pipeline.sh

2. ** in terminal: (insert your headline in quotes)
    python classify_headline.py "medical procedures are about health and hollywood"



# OR launch codes step by step:


# 1. Run Data preprocessing for model training

1. ** Execute in terminal 
    python src/data_preprocessing.py

# 2. Train model

1. ** Execute in terminal 
    python src/modeling.py


# 3. Run API

1. ** Execute in terminal 
    python app.py

2. ** POST option --> via terminal (other window - remember to have there conda env activated) any text you want to classify:
    curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"headline":"medical procedures are aabout health"}' \
    http://localhost:8000/predict

3. ** GET option --> via browser - insert any text you want to classify at the end of link after ?:
    http://0.0.0.0:8000/predict_get?headline=medical+procedures+are+about+health


# Additiaonly you can run tests:
    python -m pytest tests/ -vs