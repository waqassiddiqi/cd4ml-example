import sys
import os
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.authentication import ServicePrincipalAuthentication

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Usage: python deploy_model.py <run_id> <model_name>")
        
    train_run_id = sys.argv[1]
    model_name = sys.argv[2]

    MANDATORY_ENV_VARS = ["AZURE_ML_WORKSPACE_NAME", "AZURE_ML_SUBSCRIPTION_ID",
                          "AZURE_RESOURCE_GROUP", "AZURE_ML_TENANT_ID", 
                          "AZURE_ML_SERVICE_PRINCIPAL_ID", "AZURE_ML_SERVICE_PRINCIPAL_PASSWORD"]
    for var in MANDATORY_ENV_VARS:
        if var not in os.environ:
            raise EnvironmentError("Failed because {} is not set.".format(var))

    svc_pr = ServicePrincipalAuthentication(tenant_id=os.environ["AZURE_ML_TENANT_ID"], 
        service_principal_id=["AZURE_ML_SERVICE_PRINCIPAL_ID"], 
        service_principal_password=os.environ["AZURE_ML_SERVICE_PRINCIPAL_PASSWORD"])

    azure_workspace = Workspace.get(os.environ["AZURE_ML_WORKSPACE_NAME"], 
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"], 
        resource_group=os.environ["AZURE_RESOURCE_GROUP"]
        auth=svc_pr)

    model_path = "runs:/" + train_run_id + "/model"
    
    azure_image, azure_model = mlflow.azureml.build_image(model_uri=model_path,
                                                          workspace=azure_workspace,
                                                          model_name=model_name,
                                                          description="Model deployed using MLOps template repository",
                                                          synchronous=True)

    webservice_deployment_config = AciWebservice.deploy_configuration()
    webservice = Webservice.deploy_from_image(image=azure_image,
                                              workspace=azure_workspace, name="templatemodelservice")
    webservice.wait_for_deployment()

    print("Scoring URI is: %s", webservice.scoring_uri)
    print(f"::set-output name=scoring_uri::{webservice.scoring_uri}")
