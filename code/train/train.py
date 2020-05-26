import os
import warnings
import sys

from dotenv import load_dotenv

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from azureml.core import Run, Workspace, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(args):
    load_dotenv()
    train_on_local = os.environ.get("TRAIN_LOCAL") is not None and os.environ["TRAIN_LOCAL"] == "True"

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    if(train_on_local == True):
        ws = Workspace.get(os.environ["AZURE_ML_WORKSPACE_NAME"], 
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"], 
        resource_group=os.environ["AZURE_RESOURCE_GROUP"])

    else:
        run = Run.get_context()
        print(run)
        ws = run.experiment.workspace
        print(ws)
    
    print(mlflow.get_mlflow_tracking_uri(ws))

    # Get data from Azure ML workspace Dataset
    try:
        data = Dataset.get_by_name(workspace=ws, name="WineQualityRedDS").to_pandas_dataframe()
    except Exception as e:
        logger.exception(
            "WineQualityRedDS data set not found please check Datasets in Azure ML workspace. Error: %s", e)
    data.fillna(data.mean(), inplace=True)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # log model
        mlflow.sklearn.log_model(lr, "model")

    run_metrics = run.get_metrics(recursive=True)
    print(run_metrics)


if __name__ == '__main__':
    main(args=[])
