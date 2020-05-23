-include .env

dvc-repro:
	dvc repro train.dvc

train: 
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && mlflow run --experiment-name "${MLFLOW_EXPRIMENT_NAME}" .

evaluate:
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && python src/evaluate.py ${MLFLOW_TRAIN_RUN_ID}

register-model:
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && python src/register_model.py ${MLFLOW_TRAIN_RUN_ID} ${MLFLOW_MODEL_NAME}