# {{deployment_name}}

This is the main deployment repository for the `{{deployment_name}}` of the `{{registered_model_name}}` model using LUME-Model in an online environment.

## Overview

This repository was created from the [lume-model-deployment-template](https://github.com/slaclab/lume-model-deployment-template) using `copier`. 
It provides a structured approach to deploying machine learning models using [LUME-Model](https://github.com/slaclab/lume-model) 
in an online environment. It offers a reproducible method for containerizing and deploying ML models, ensuring consistency
and ease of use across different projects, while minimizing boilerplate code for deployment.

Please refer to the [original template repository](https://github.com/slaclab/lume-model-deployment-template) for detailed 
documentation on its features and usage. 
If the template is updated and you want to apply changes to your project:
```bash
copier update
```
This will re-apply the template, preserving your answers and customizations where possible.

## How to Use

### 0. Register Your Model and Prepare PV Mapping
Before using this template, ensure you have registered your model in MLflow and prepared the PV mapping for your deployment.

#### Register the model
Assuming you have a trained model wrapped in a LUME model class (e.g., `TorchModel`, `SklearnModel`, etc.), you can register it to MLflow as shown
below. 

Note that: 
- **you must edit the metadata section for your specific model**
- **you must have MLflow installed**
- **you must be on the SLAC network to register to the SLAC MLflow registry**

```python
import mlflow
from lume_model.models import TorchModel

# load model from yaml
model = TorchModel("model_config.yaml")

## Editable Configuration 
MLFLOW_URI = "https://ard-mlflow.slac.stanford.edu"
EXPERIMENT_NAME = "lume-example-deployment"

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Metadata - EDIT FOR EACH MODEL
# Only use lowercase alphanumeric characters and hyphens!
REGISTERED_MODEL_NAME = "lume-example-model"  

EMAIL = "user@slac.stanford.edu"
REPO = "https://github.com/slaclab/lume-example-model"
BEAM_PATH = "cu_hxr"
DESCRIPTION = "Example model"
READY_TO_DEPLOY = "true"  # Set to "true" for production (sends to controls servers), "false" for dev
STAGE = "production"  # development, staging, or production

# Set tags
model_tags = {
    "email": EMAIL,
    "repo": REPO,
    "beam_path": BEAM_PATH,
    "description": DESCRIPTION,
    "stage": STAGE,
}

version_tags = {
    "ready_to_deploy": READY_TO_DEPLOY,
}

# Register model
model.register_to_mlflow(
    artifact_path="model",
    registered_model_name=REGISTERED_MODEL_NAME,
    tags=model_tags,
    version_tags=version_tags
)
```

#### Generate PV Mapping

You need to create a file called `pv_mapping.yaml` that defines how the EPICS PVs map to the model's input and
output features. This file will later be copied into the template project.

The structure of the `pv_mapping.yaml` file should look like this example:

```yaml
input_variables:
  model_input_1:
    symbols:
      - INPUT:PV:1
      - INPUT:PV:2
    formula: (INPUT:PV:1**2 + INPUT:PV:2**2)**(1/2)
    proto: ca
  model_input_2:
    formula: "1.850"  # constant value needs to match lume-model config exactly
  model_input_3:
    symbols:
      - INPUT:PV:3
    formula: INPUT:PV:3
    proto: ca
    ...
output_variables:
  OUTPUT:PV:1:
    symbols: 
      - model_output_1
    formula: model_output_1 * 2.0 + 1.0
    proto: pva
    ...
```

Note that constants must be set correctly in your config; mismatches may cause validation errors.

> [!IMPORTANT]  
> There is currently no validation implemented for this; user must ensure config matches the LUME-model and that the mapping
is defined correctly.


### 1. Create a Deployment Repository

In a Python environment with `copier` installed, run:

```bash
mkdir lume-example-deployment
copier copy gh:slaclab/lume-model-deployment-template lume-example-deployment
cd lume-example-deployment
```

### 2. Add Your PV Mapping

Copy the `pv_mapping.yaml` file that you created in step 0 with your PV names and how they map to the model features
to the `src/online_model/configs/` directory. If you want output PVs to be written back to EPICS,
ensure that the output PVs are included in this mapping as well.

### 3. Initialize and Push to GitHub

Create a new repository on GitHub under the `slaclab` org (e.g., `lume-example-deployment`). Note that the repository
must be public, otherwise, additional configuration is needed (configuring tokens/authorizations).

Then run:

```bash
git init
git add -A
git commit -m "init commit"
git remote add origin https://github.com/slaclab/lume-example-deployment.git
git push --set-upstream origin main
```

Once pushed, this will automatically trigger a GitHub Actions workflow to build and push the Docker image to
the GitHub Container Registry under `slaclab`. Once that's done, you can deploy to your target Kubernetes cluster.
If ArgoCD is already set up, it will automatically deploy the new image.


## Optional Local Testing
If you want to test the deployment locally before pushing to GitHub, you can build and run the Docker image locally.
Make sure you either are on SLAC network or have access to the MLflow server you are using, and that you have Docker installed.

> [!IMPORTANT]
> The "test" interface does not connect to EPICS and does not use the pv_mapping or any EPICS related
> transforms or config.  It does not test any I/O
> interface, but instead generates random values for each input variable from their specified ranges. 
> Therefore, the output values are not meaningful, but this allows you to test the most of the inference run.

To build and run the Docker image locally with the `test` interface, run:
```bash
docker build --build-arg INTERFACE=test -t lume:test .
docker run -e INTERFACE=test -e MODEL_VERSION=1 lume:test
```

You can also access the container's shell with:
```bash
docker run -it lume:test bash
```

If you want to test with a local MLflow server, you can run in your terminal with MLflow installed and use whatever port
is available (e.g., 8082) (see [docs](https://mlflow.org/docs/latest/getting-started/intro-quickstart/)):
```bash 
mlflow server --host 127.0.0.1 --port 8082 --gunicorn-opts --timeout=60
````

Then edit the `template_config` to set `mlflow_tracking_uri="http://127.0.0.1:8082"`.

---

For more details, see the [lume-model documentation](https://github.com/slaclab/lume-model) and the [Copier documentation](https://copier.readthedocs.io/).

