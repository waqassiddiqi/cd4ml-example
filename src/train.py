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

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    load_dotenv()
    MANDATORY_ENV_VARS = ["MLFLOW_TRACKING_URI"]
    for var in MANDATORY_ENV_VARS:
        if var not in os.environ:
            raise EnvironmentError("Failed because {} is not set.".format(var))
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read output.csv data
    try:
        data = pd.read_csv("data/output.csv", sep=',')
    except Exception as e:
        logger.exception(
            "output.csv file not found, run dvc repro preprocess.dvc first. Error: %s", e)

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

        # set current run id for next step of pipleline
        print(mlflow.active_run().info.run_id)
        print(f"::set-output name=run_id::{mlflow.active_run().info.run_id}")
