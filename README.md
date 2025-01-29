# MLE Document Classifier 

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

# Steps:

## 0. Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Python 3.8+**: You can download it from [python.org](https://www.python.org/downloads/).
2. **Git**: You can download it from [git-scm.com](https://git-scm.com/downloads).
3. **Docker**: You can download it from [docker.com](https://www.docker.com/get-started).
4. **wget**: For downloading files from the web (optional, can download manually).

For best results, use Linux or macOS. If you are using Windows, consider using Windows Subsystem for Linux (WSL) to create a Linux-like environment.

Make sure to have these tools installed and properly configured before proceeding with the steps below.
## 1. Clone the Repository:

    git clone https://github.com/mnowicki94/documents_classifier_api.git
    cd https://github.com/mnowicki94/documents_classifier_api.git

## 2. Download data

    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
    unzip NewsAggregatorDataset.zip -d data

    OR just download manually from https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
    and copy to data folder in your repo

    alternatively in Powershell:
    Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip" -OutFile "NewsAggregatorDataset.zip"
    Expand-Archive -Path "NewsAggregatorDataset.zip" -DestinationPath "data"



## 3 Setting Up the Environment
### a. via venv (recommended)

    python -m venv env && source env/bin/activate && pip install -r requirements.txt


    alternatively in Powershell:
    python -m venv env
    env/Scripts/activate
    pip install -r requirements.txt


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

#### b.i write titles to classify in terminal (space between titels)
    
    python classify_headlines.py "medical procedures are about health" "robots are coming"

#### b.ii use titles to classify from file
    
    python classify_headlines.py --file titles_to_test.txt

### c. Run via endpoint

    http://0.0.0.0/predict_get?headline=your_text → Predict category

