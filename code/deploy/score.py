import os
import pandas as pd

from azureml.core.model import Model

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import parse_json_input, _get_jsonable_obj


def init():
    print(os.environ)
    print(os.listdir("."))
    global model


def run(json_input):
    input_df = parse_json_input(json_input=json_input, orient="split")
    return _get_jsonable_obj(model.predict(input_df), pandas_orient="records")
