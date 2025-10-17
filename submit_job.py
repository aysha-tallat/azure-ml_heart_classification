from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Job
import os

# -----------------------------
# 1️⃣ Connect to your workspace
# -----------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<your subsription_id>",  
    resource_group_name="rg-dp100",
    workspace_name="mlws-dp100"
)

# -----------------------------
# 2️⃣ Define the training job
# -----------------------------
job = command(
    code="./",                                  # your code folder (current directory)
    command="python train.py --input_data ${{inputs.input_data}}",                 # command to run inside the environment
    inputs={
        "input_data": Input(type="uri_file", path="azureml:heart-data:1"),
    },
    environment="azureml:heart-env:1",          # existing environment
    compute="aml-cluster",                      # existing compute target
    display_name="heart-train-job",
    description="Train heart disease classification model using Azure ML SDK v2",
    experiment_name="heart_experiment"
)

# -----------------------------
# 3️⃣ Submit the job
# -----------------------------
returned_job = ml_client.jobs.create_or_update(job)

# -----------------------------
# 4️⃣ Print job info
# -----------------------------
print(f"Job submitted successfully! Job name: {returned_job.name}")