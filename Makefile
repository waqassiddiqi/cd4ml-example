include .env

dvc-repro:
	dvc repro train.dvc

run: 
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && mlflow run --experiment-name "${MLFLOW_EXPRIMENT_NAME}" .

serve:
	mlflow models serve -m ./model -p 9001