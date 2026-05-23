import sys
import os
import argparse
import time

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
)
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError

def main():
    parser = argparse.ArgumentParser(
        description="Deploy the Retail Oracle custom model to Azure ML Managed Online Endpoint."
    )
    parser.add_argument("--subscription_id", default="4d71fa38-49e9-45c1-8a01-534eb7e98d3f", help="Azure Subscription ID")
    parser.add_argument("--resource_group", default="ProjectMLInstacart", help="Azure Resource Group Name")
    parser.add_argument("--workspace_name", default="instacart-ws", help="Azure ML Workspace Name")
    parser.add_argument(
        "--endpoint_name",
        default="instacart-endpoint",
        help="Endpoint name",
    )
    parser.add_argument(
        "--instance_type",
        default="Standard_DS2_v2",
        help="Azure VM instance type (DS2_v2 fits 4-core limit)",
    )
    parser.add_argument(
        "--deployment_name",
        default="blue",
        help="Deployment name",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Connect to Azure ML ─────────────────────────────────────────────────
    print("Connecting to Azure ML Workspace...")
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name,
        )
        print(f"Connected to workspace: {ml_client.workspace_name}")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)

    # ── 1. Register Custom Model ────────────────────────────────────────────
    print("Registering custom pickle model in workspace...")
    model_path = os.path.abspath(os.path.join(script_dir, "..", "model_mlflow"))
    if not os.path.exists(os.path.join(model_path, "model.pkl")):
        print(f"[ERROR] model.pkl not found in {model_path}")
        sys.exit(1)

    model_entity = Model(
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name="retail-oracle-custom",
        description="Retail Oracle Custom Pickled Model Package flat",
    )
    registered_model = ml_client.models.create_or_update(model_entity)
    print(
        f"Model registered -> name: {registered_model.name}, "
        f"version: {registered_model.version}, id: {registered_model.id}"
    )

    # ── 2. Register Custom Environment ──────────────────────────────────────
    print("Registering custom environment in workspace...")
    conda_path = os.path.join(script_dir, "conda_env.yml")
    if not os.path.exists(conda_path):
        print(f"[ERROR] conda_env.yml not found at {conda_path}")
        sys.exit(1)

    env = Environment(
        name="retail-oracle-env",
        description="Custom environment for Retail Oracle with dependencies",
        conda_file=conda_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )
    registered_env = ml_client.environments.create_or_update(env)
    print(
        f"Environment registered -> name: {registered_env.name}, "
        f"version: {registered_env.version}, id: {registered_env.id}"
    )

    # ── 3. Ensure Managed Online Endpoint exists ──────────────────────────
    print(f"Checking/Creating endpoint '{args.endpoint_name}'...")
    try:
        endpoint = ml_client.online_endpoints.get(args.endpoint_name)
        print(f"Endpoint '{args.endpoint_name}' already exists.")
    except ResourceNotFoundError:
        print(f"Endpoint '{args.endpoint_name}' not found. Creating it...")
        endpoint = ManagedOnlineEndpoint(
            name=args.endpoint_name,
            description="Retail Oracle Predictive Cart Custom - Managed Online Endpoint",
            auth_mode="key",
        )
        ml_client.begin_create_or_update(endpoint).result()
        print(f"Endpoint '{args.endpoint_name}' created.")

    # ── 4. Deploy Custom Model ──────────────────────────────────────────────
    print(f"Deploying model to endpoint '{args.endpoint_name}'...")
    print(f"Code configuration: dir={script_dir}, script=score.py")

    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=registered_model.id,
        environment=registered_env.id,
        code_configuration=CodeConfiguration(
            code=script_dir,
            scoring_script="score.py",
        ),
        instance_type=args.instance_type,
        instance_count=1,
    )
    
    print("Starting deployment (this may take several minutes)...")
    ml_client.begin_create_or_update(deployment).result()
    print("Deployment completed successfully!")

    # ── 5. Route 100% traffic to this deployment ───────────────────────────
    print(f"Routing 100% traffic to '{args.deployment_name}'...")
    endpoint.traffic = {args.deployment_name: 100}
    ml_client.begin_create_or_update(endpoint).result()
    print("Traffic routed successfully.")

    # ── 6. Run quick smoke test ────────────────────────────────────────────
    sample_path = os.path.join(script_dir, "sample_request.json")
    if os.path.exists(sample_path):
        print("Running smoke test against the live endpoint...")
        try:
            response = ml_client.online_endpoints.invoke(
                endpoint_name=args.endpoint_name,
                request_file=sample_path,
            )
            print("Endpoint response:")
            print(response)
        except Exception as e:
            print(f"Smoke test failed: {e}")

if __name__ == "__main__":
    main()
