# Datascientest project - Analysis of Covid-19 chest x-rays

## Project description

In order to diagnose patients with Covid-19, the analysis of chest X-rays is a possibility to be explored to more easily detect positive cases. If the classification through deep learning of such data proves effective in detecting positive cases, then this method can be used in hospitals and clinics when traditional testing cannot be done.

## Resources to refer to:
### Data:
The data set contains chest x-ray images for covid-19 positive cases but also x-ray images of normal and viral pneumonia. Link to dataset: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
(Data size: 1.15 Gb)

### Bibliography:
https://arxiv.org/abs/2003.13865
https://doi.org/10.1016/j.compbiomed.2021.105002


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------
## Running the project
### GIT
Git commands to use in terminal/console
#### show git log
```bash
git log --oneline --graph --name-status
```
#### show changes by user (blame)
```bash
git blame filename
```
#### show change between commits
```bash
git diff commit1 commit2
```
#### reset change to last commit
```bash
git reset --soft HEAD~1
```
#### create new branch
```bash
git checkout -b new-branch-name
```
#### add stages files
```bash
git add file1
```
#### commit changes with a message
```bash
git commit -m "message"
```
### Python virtual environment
To setup venv for this project, use the script `venv.sh` in the root folder of this project as the following:
```bash
source venv.sh help
```
### Install project dependencies
To install project's dependencies, use Makefile in the root folder of this project as the following:
```bash
make install
```
### Jupyter notebook kernel to the .ds_covid19 env
To set Jupyter's notebook kernel to .ds_covid19 env, execute this command after the env is activate as the following:
```bash
python3 -m ipykernel install  --user --name=ds_covid19
```
### Pre-commit for python
To check that all your files follow proper standards, execute this command manually as the following:
```bash
pre-commit run --all-files
```
### Run MLFlow
To start MLFlow, execute this command manually as the following:
```bash
make mlflow-start
```
To stop MLFlow, execute this command manually as the following:
```bash
make mlflow-stop
```
### Run Streamlit
To run Streamlit, execute this command manually as the following:
```bash
streamlit run src/streamlit/Home.py
```

### Model training and testing
To train the model, you can either follow in the instructions in 5.5 notebook of execute the following commands:
```bash
make train
```

To test the model, you can either follow in the instructions in 5.5 notebook of execute the following commands:
```bash
make test
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
