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


## 1.Clone the Repository:

    git clone https://github.com/mnowicki94/documents_classifier_api.git
    cd https://github.com/mnowicki94/documents_classifier_api.git

## 2. Download data

    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
    unzip NewsAggregatorDataset.zip -d data

## 3 Setting Up the Environment
### a. via venv (recommended)

    python3 -m venv env && source env/bin/activate && pip install -r requirements.txt

### b. via conda

    conda env create -f environment.yml
    conda activate ml_doc_classifier_env


## 4. Development process: model training
### Run data_preprocessing.py and modeling.py via script below

    chmod +x ./run/run_model_training.sh
    ./run/run_model_training.sh


## 5. Production Environment: API Serving
### a. Run docker running container launching app.py that creates API that serves models from model training

    chmod +x ./run/run_api.sh
    ./run/run_api.sh

### b. Run classify_headlines.py to use the API

#### b.a write titles to classify in terminal (space between titels)
    
    python classify_headlines.py "medical procedures are about health" "robots are coming"

#### b.b. use titles to classify from file
    
    python classify_headlines.py --file titles_to_test.txt

