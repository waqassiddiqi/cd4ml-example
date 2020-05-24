import mlflow.sklearn
import pandas as pd
import os
import sys
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Run ID is missing")

    train_run_id = sys.argv[1]

    try:
        data = pd.read_csv("data/output_validate.csv", sep=',')
    except Exception as e:
        logger.exception(
            "output.csv file not found, run dvc repro preprocess.dvc first. Error: %s", e)

    model = mlflow.sklearn.load_model("runs:/" + train_run_id + "/model")

    # test data
    test_quality = data.quality
    test = data.drop(["quality"], axis=1)

    # predictions
    predictions = model.predict(test)
    (rmse, mae, r2) = eval_metrics(test_quality, predictions)
    
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # check model passess baseline
    if rmse < 0.5:
        raise ValueError("RMSE {} is lower than baseline {}".format(rmse, 0.5))
    
