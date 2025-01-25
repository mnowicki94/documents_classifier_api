# Machine Learning Engineer Take-Home Exercise

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

Your model does not need to use all attributes. You are free to determine which ones to use and which to discard (using analysis or your own judgment). It’s also ok to take a data sample instead of using the entire data set.

## Report
- **Code**: You can include a notebook for your exploratory steps, but the final output should also include runnable Python code as a Python file or module.
- **Presentation**: PowerPoint, etc.

## Evaluation Criteria
Performance of the final model is not a very important criterion for this exercise; we are more interested in:
- The step-by-step approach and thought process.
- The soundness and robustness of the solution and evaluation metrics.
- The quality and clarity of the code. Your Python code should be well formatted and (for example) contain docstrings.
- Considerations for possible next steps are also welcome (for example if there are things you would have liked to try if you had had more time).




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