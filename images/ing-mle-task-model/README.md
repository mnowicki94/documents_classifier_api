# ING MLE Task Model

## IMPORANT
Input data is not part of the zip package.
Need to be placed in input folder:

ing_mle_task_model/
├── input/
│   └── newsCorpora.csv

## Prerequisities 
Python 3.8.7 and optionally poetry installed

## Installation
To set up the project environment, follow these steps:

Create virtual environment using poetry and pyproject.toml:

```bash
poetry install
```


Using venv and requirements.txt file:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## pytest

To run all tests included in repository run:

    pytest