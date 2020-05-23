-include .env

dvc-repro:
	dvc repro train.dvc

train: 
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && mlflow run --experiment-name "${MLFLOW_EXPRIMENT_NAME}" .

evaluate:
	export export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && mlflow run --experiment-name "${MLFLOW_EXPRIMENT_NAME}" -e evaluate -P run_id=${MLFLOW_TRAIN_RUN_ID} .