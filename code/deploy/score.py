import os
import pandas as pd

from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import parse_json_input, _get_jsonable_obj


def init():
   global model
   model = load_model(os.environ.get("AZUREML_MODEL_DIR"))

@input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4]])))
@output_schema(StandardPythonParameterType({'predict': [['Iris-virginica']]}))
def run(json_input):
    input_df = parse_json_input(json_input=json_input, orient="split")
    result = _get_jsonable_obj(model.predict(input_df), pandas_orient="records")

    inputs_dc.collect(input_df)
    prediction_dc.collect(result)

    return result
