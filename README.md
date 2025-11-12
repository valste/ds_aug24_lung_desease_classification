# Datascientest project - Analysis of Covid-19 chest x-rays

## My solo work is presented in the following notebook files, along with all related artifacts and sources used within them, as well as the dockerization setup.
    notebooks: 
    * 1.1_Explore_metadata.ipynb
    * 2.0_Estimate_image_orientation.ipynb
    * 5.6_Capsnet_4class_disease_classifier.ipynb
 

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
    ├── models(isolated huggingface repos)<- Trained and serialized models, model predictions, or model summaries
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




 ## How to setup each models/ as its own Huggungface repo with its own remote, fully isolated from the parent?

 > 📝 **Notes:**
>
> * Each directory inside `models/` is its **own Git repository**, pushed to **Hugging Face**.
> * The parent repository’s `.gitignore` excludes `models/*`, so these are **not** tracked or pushed to GitHub.
> * You can commit, push, and version each model independently using its Hugging Face remote.

### 🧩 Clone Hugging Face Model Repositories


```bash
# Navigate to your parent project root
cd <parent-root>

# Ensure the models folder exists (GitHub ignores it except for .gitkeep)
mkdir -p models
touch models/.gitkeep

# Clone each Hugging Face child repo into its subfolder
git clone https://huggingface.co/<user>/<capsnet-repo> models/capsnet-4class-lung-disease-classifier
git clone https://huggingface.co/<user>/<mobnet-repo> models/orientation-classifier-224x224-aug-head1-mobnet
git clone https://huggingface.co/<user>/<resnet-repo> models/orientation-classifier-224x224-aug-head1-resnet50
git clone https://huggingface.co/<user>/<gan-repo> models/lung-segmentation-gan
git clone https://huggingface.co/<user>/<unet-repo> models/lung-segmentation-unet
git clone https://huggingface.co/<user>/<covid19-repo> models/ds-cxr-covid19
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
