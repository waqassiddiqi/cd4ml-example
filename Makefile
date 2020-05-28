-include .env

dvc-repro-preprocess:
	dvc repro preprocess.dvc

train: 
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && mlflow run --experiment-name "${MLFLOW_EXPRIMENT_NAME}" .

evaluate:
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && mlflow run --experiment-name "${MLFLOW_EXPRIMENT_NAME}" --entry-point evaluate --param-list run_id=${MLFLOW_TRAIN_RUN_ID} .

register-model:
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && python src/register_model.py ${MLFLOW_TRAIN_RUN_ID} ${MLFLOW_MODEL_NAME}