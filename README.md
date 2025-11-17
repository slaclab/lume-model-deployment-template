# LUME-Model Deployment Template

> [!CAUTION]
> This template is under active development and may change. Please report any issues or suggestions.

This repository provides a [copier](https://copier.readthedocs.io/) template for deploying machine learning models 
using [LUME-Model](https://github.com/slaclab/lume-model) in an online environment. It offers a structured, 
reproducible approach to containerizing and deploying ML models, ensuring consistency and ease of use across different
projects, and reducing the overhead of writing boilerplate code for deployment.

It was developed for use at SLAC National Accelerator Laboratory, particularly for deploying models on the S3DF
Kubernetes cluster, but can be adapted for other environments. It supports continuous inference using inputs from EPICS
PVs, and optionally writing outputs back to EPICS PVs. It also supports MLflow for model versioning and tracking.

<div align="center">
  <img src="docs/assets/mlops_inference_diagram.png" alt="MLOps Deployment Diagram" width="600"/>
</div>

> [!IMPORTANT]  
> Before creating a deployment project using this template, please ensure that:
> * your model is logged in [mlflow](https://mlflow.org/) and is versioned and tagged appropriately
> * your model is trained and has the appropriate transformations to work with raw machine data through LUME-model
> * you want to deploy your model on an S3DF Kubernetes cluster where it will continuously run inference using
inputs from EPICS PVs, and optionally write outputs back to EPICS PVs

---

## Use Cases
- **Accelerator controls**: Deploy ML models for online inference in accelerator control systems.
- **Rapid prototyping**: Quickly scaffold new model deployment projects with best practices.
- **Consistent deployment**: Standardize deployment across teams and projects.

---

## Getting Started

### Prerequisites
- [Python 3.10+](https://www.python.org/)
- [copier](https://copier.readthedocs.io/en/stable/) (`pip install copier`)

### Installation
Install copier if you haven't already:
```bash
pip install copier
```

---

## Using the Template

To generate a new deployment project, create a new folder for the project. We recommend giving your 
directory/repository a meaningful name related to your deployment, such as `snd-nn-model-deployment`. Then run:
```bash
copier copy gh:slaclab/lume-model-deployment-template <destination-folder>
```
You will be prompted for configuration values (e.g., deployment name, model version, etc.). You should have the
following information ready:
- **registered_model_name**: Name of the model as registered in the MLflow Model Registry. This should be something 
like `snd-nn-model` or `lcls-cu-inj-model`.
- **model_version**: Version of the model as registered in the MLflow Model Registry. This should be an integer, e.g., `1`.
- **deployment_name**: Name of the Kubernetes deployment. This must be unique within the Kubernetes namespace you are deploying to.
It should be something like `snd-nn-model-deployment` or `lcls-cu-inj-model-deployment`.
**Note that all deployments will be in the same namespace, `lume-online-ml`, so choose unique names.**
- **rate**: Inference rate in Hz. This is how often the model will run inference. Default is `1` (once per second).
- **extra_pip_requirements**: Any additional pip packages your model needs that are not already included in the base image.
- **mlflow_tracking_uri**: The MLflow tracking server URI. This can be either the production server or a local server for testing.
- **vcluster**: The name of the vcluster to deploy to. This should be either `ad-accel-online-ml` or `lcls-ml-online`.
- **interface**: The interface to use for EPICS PV access. This should be either `epics`, `k2eg`, or `test`.

### Updating an Existing Deployment
If the template is updated and you want to apply changes to your project:
```bash
copier update
```
This will re-apply the template, preserving your answers and customizations where possible.

---

## Project Structure
- `src/` — Source code for deployment logic and interfaces
- `config/` — Configuration files for deployments
- `Dockerfile.jinja` — Jinja-templated Dockerfile for containerization
- `pixi.toml` — Python dependencies
- `deployment.yaml` — Jinja-templated deployment configuration

---

# Deploying After Generating Your Project

1. Navigate to your new project directory:
   ```bash
   cd <destination-folder>
   ```
2. Review and customize the generated files as needed.
**Note that it is not recommended to change the template or source code to avoid complications when updating the template in the future,
and to ensure consistency across deployments.**
3. Test your deployment locally or in your target environment following the instructions in the generated `README.md`.
4. Once satisfied, push your project to GitHub under `slaclab` (recommended for S3DF deployments). We recommend giving your 
directory/repository a meaningful name related to your deployment, such as `snd-nn-model-deployment` or `lcls-cu-inj-model-deployment`.
5. A GitHub Actions workflow is already configured for your repository. It will build and push your Docker image to 
the GitHub Container Register under `slaclab` upon pushing to the `main` branch, and update the deployment file with the 
latest image tag.
6. For the initial deployment, you will need someone with access to the Kubernetes cluster to help set up the deployment.
Once your project repo is set up, create a new issue in your repo and tag the team for assistance.
7. After the initial setup, you can obtain vcluster access and manage and update your deployment using `kubectl` commands.
8. Update the deployment as needed using `copier update`. 


Don't forget to document any customizations or changes made to the template, and share feedback or contribute 
improvements to the template repository!

---
# Development Roadmap

## 1-2 months
- [ ] Add support for pre-model-evaluation PV transformations (e.g., formulas)
- [ ] Add support for local model loading for models that are not in MLflow
- [ ] Expand docs (how to upload models to MLflow and versioning/tagging requirements, how to set up MLflow locally 
for testing, how to set up vcluster access, etc)
- [ ] Tutorial + demo for 1-2 use cases

## 2-4 months
- [ ] Add templated way of adding requirements from the user at project generation
- [ ] Add support for writing outputs back to EPICS PVs
- [ ] Add support for automated deployment to S3DF Kubernetes (tentative)
- [ ] Add support for gpu-based models
