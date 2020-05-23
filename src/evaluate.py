import mlflow.sklearn
import pandas as pd
import os
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    load_dotenv()
    MANDATORY_ENV_VARS = ["MLFLOW_RUN_ID"]

    try:
        data = pd.read_csv("data/output.csv", sep=',')
    except Exception as e:
        logger.exception(
            "output.csv file not found, run dvc repro preprocess.dvc first. Error: %s", e)

    model = mlflow.sklearn.load_model("runs:/" + os.environ["MLFLOW_RUN_ID"] + "/model")

    # test data
    test_quality = data.quality
    test = data.drop(["quality"], axis=1)

    # predictions
    predictions = model.predict(test)
    (rmse, mae, r2) = eval_metrics(test_quality, predictions)

    # check model passess baseline
    if rmse < 0.5:
        raise ValueError("RMSE {} is lower than baseline {}".format(rmse, 0.5))
    
