#---------------------------------------------------
# Project venv and kernel
#---------------------------------------------------
# Virtual environment name:
VENV = .ds_covid19

# Install project's dependencies
install:
	pip install -r requirements.txt -U

# Store install packages for venv in requirements.txt
freeze:
	pip freeze > requirements.txt

#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------
# Preprocess the data
preprocess:
	python -m src.preprocess.build_features

# Train the model
train:
	python -m src.models.train_model

# Make predictions on the test data
test:
	python -m src.models.predict_model

# Create reports
html-reports:
	echo "Creating reports..."
	find reports/ -type f -name "*.html" -delete
	jupyter nbconvert --to html notebooks/*.ipynb --output-dir=reports

# Evaluate performance
evaluate:
	python -m src.evaluate.evaluate

# Produce visualizations
visualize:
	python -m src.visualization.visualize

# Run all: RUNS ALL SCRIPTS - DEFAULT
all: download preprocess train test evaluate visualize

#---------------------------------------------------
# Running unit tests
#---------------------------------------------------
## Run all tests
unit-tests:
	pytest

#---------------------------------------------------
# Running MLFlow server locally
#---------------------------------------------------
## Start MLFLow server
mlflow-start:
	@echo "Start MLflow Server..."
	mlflow server --host 127.0.0.1 --port 8080 &
	@echo "MLflow Server Started!"

mlflow-stop:
	@echo "Stopping MLflow Server..."
	@pkill -f "gunicorn" || echo "MLflow is not running."
	@echo "MLflow Server Stopped!"
#---------------------------------------------------
# Cleaning folders
#---------------------------------------------------
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Delete all data
clean-data:
	rm -rf data/raw/*
	rm -rf data/interim/*
	rm -rf data/processed/*

# Delete all models, metrics, and visualizations
clean-results:
	rm -rf model/*
	rm -rf results/*
	rm -rf reports/figures/*

# Delete all
clean-all: clean clean-data clean-results
