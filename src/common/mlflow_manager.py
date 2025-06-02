# -*- coding: utf-8 -*-
import io
import json
import os

import mlflow
import regex as re
from mlflow.models import infer_signature


def log_model(
    expermint,
    framework,
    description,
    model,
    model_name,
    test_data,
    metrics,
    classification_type,
    parameters=None,
    history=None,
):
    """
    Log a model to MLflow

    input:
    experiment: The experiment to log the model to
    framework: The framework of the model
    description: The description of the dataset
    model: The model to log
    model_name: The name of the model
    test_data: The test data to infer the signature
    metrics: The metrics to log
    classification_type: The type of classification
    parameters: The parameters to log
    history: The training history of the model (optional)

    output:
    model_info: The model info
    """

    # Create a unique run name
    run_name = f"{model_name}-{classification_type}"

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://localhost:8080")

    # Create a new MLflow Experiment
    experiment_tags = {
        "project_name": "ds_covid19",
        "course_term": "Aug24_CDS",
        "team": "stores-ml",
        "project_period": "11-2024_06-2025",
        "mlflow.note.content": "Datascientest project - Analysis of Covid-19 chest x-rays",
    }

    mlflow.set_experiment(expermint)
    mlflow.set_experiment_tags(experiment_tags)

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):

        # Set a tag that we can use to remind ourselves what this run was for
        run_tags = {
            "mlflow.note.content": f"{description} using {framework}-{model_name}",
            "Classification type": f"{classification_type}",
            "Classification dataset": re.search(r"\[(.*?)\]", description).group(),
        }

        mlflow.set_tags(run_tags)

        # Log the loss metric
        mlflow.log_metrics(metrics)

        # Log the parameters
        # mlflow.log_params(parameters)

        signature = infer_signature(test_data, model.predict(test_data))

        match framework:
            case "tensorflow":
                stream = io.StringIO()
                model.summary(print_fn=lambda x: stream.write(x + "\n"))
                summary_str = stream.getvalue()

                with open("model_summary.txt", "w") as f:
                    f.write(summary_str)

                mlflow.log_artifact("model_summary.txt")
                os.remove("model_summary.txt")

                if history:
                    with open("training_history.json", "w") as f:
                        json.dump(history.history, f)

                    mlflow.log_artifact("training_history.json")
                    os.remove("training_history.json")

                # Log the model
                model_info = mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path=model_name,
                    signature=signature,
                    registered_model_name=f"{framework}-{model_name}-{classification_type}",
                )
            case "sklearn":

                # Log the model
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    registered_model_name=f"{framework}-{model_name}-{classification_type}",
                )
            case "keras":
                # Log the model
                model_info = mlflow.keras.log_model(
                    keras_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    registered_model_name=f"{framework}-{model_name}-{classification_type}",
                )
            case _:
                raise TypeError("Framework not supported")

    return model_info
