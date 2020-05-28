import sys
import os
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise ValueError("Usage: python deploy_model.py <run_id> <model_name>")
        
    train_run_id = sys.argv[1]
    model_name = sys.argv[2]

    MANDATORY_ENV_VARS = ["AZURE_ML_WORKSPACE_NAME", "AZURE_ML_SUBSCRIPTION_ID",
                          "AZURE_RESOURCE_GROUP"]
    for var in MANDATORY_ENV_VARS:
        if var not in os.environ:
            raise EnvironmentError("Failed because {} is not set.".format(var))

    azure_workspace = Workspace.create(name=os.environ["AZURE_ML_WORKSPACE_NAME"],
                                       subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
                                       resource_group=os.environ["AZURE_RESOURCE_GROUP"],
                                       location="southeastasia",
                                       create_resource_group=False,
                                       exist_okay=True)

    model_path = "runs:/" + train_run_id + "/model"
    
    azure_image, azure_model = mlflow.azureml.build_image(model_uri=model_path,
                                                          workspace=azure_workspace,
                                                          description=model_name,
                                                          synchronous=True)

    webservice_deployment_config = AciWebservice.deploy_configuration()
    webservice = Webservice.deploy_from_image(image=azure_image,
                                              workspace=azure_workspace, name="TemplateModelDeployment")
    webservice.wait_for_deployment()

    print("Scoring URI is: %s", webservice.scoring_uri)
    print(f"::set-output name=scoring_uri::{webservice.scoring_uri}")
