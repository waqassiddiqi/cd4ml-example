import os
import sys

import mlflow

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Usage: python register_model.py run_id model_name ")

    train_run_id = sys.argv[1]
    model_name = sys.argv[2]

    result = mlflow.register_model("runs:/" + train_run_id + "/model", model_name)
    print(result)