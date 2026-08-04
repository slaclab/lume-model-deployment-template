# LUME Online ML Surrogate Model Deployment Architecture

## Overview

This document describes the end-to-end architecture for deploying LCLS surrogate ML models as online services on Kubernetes. The system enables real-time inference from trained PyTorch models, reading live EPICS PVs and publishing predictions back to the control system.

---

## Surrogate Models

### 1. LCLS Cu Injector Surrogate (`lcls_cu_injector_ml_model`)

| Field | Details |
|-------|---------|
| **Repository** | [lcls_cu_injector_ml_model](https://github.com/slaclab/lcls_cu_injector_ml_model) |
| **Training Data** | IMPACT-T simulations (trained by Auralee Edelen) |
| **Framework** | PyTorch, wrapped with [lume-torch](https://github.com/lume-science/lume-torch) `TorchModel` |
| **Deployment Repo** | [lcls-cu-inj-model-deployment](https://github.com/slaclab/lcls-cu-inj-model-deployment) |
| **Inference Service** | `inference-service-cu-inj` (FastAPI, port 8000) |
| **MLflow Model Name** | `lcls-cu-inj-model` |
| **MLflow Version** | 1 |
| **Device** | CPU |

**Inputs (16 variables):**

| PV Name | Description | Notes |
|---------|-------------|-------|
| `SOLN:IN20:121:BACT` | Solenoid | CA protocol |
| `QUAD:IN20:121:BACT` | Quadrupole | CA |
| `QUAD:IN20:122:BACT` | Quadrupole | CA |
| `QUAD:IN20:361:BACT` | Quadrupole | CA |
| `QUAD:IN20:371:BACT` | Quadrupole | CA |
| `QUAD:IN20:425:BACT` | Quadrupole | CA |
| `QUAD:IN20:441:BACT` | Quadrupole | CA |
| `QUAD:IN20:511:BACT` | Quadrupole | CA |
| `QUAD:IN20:525:BACT` | Quadrupole | CA |
| `ACCL:IN20:300:L0A_PDES` | L0A phase | CA |
| `ACCL:IN20:400:L0B_PDES` | L0B phase | CA |
| `CAMR:IN20:186:XRMS` | Beam X RMS | CA |
| `CAMR:IN20:186:YRMS` | Beam Y RMS | CA |
| `CAMR:IN20:186:R_DIST` | Computed: √(XRMS² + YRMS²) | Derived |
| `Pulse_length` | Constant: 1.855 | Fixed from training |
| `FBCK:BCI0:1:CHRG_S` | Charge, constant: 0.25 | Fixed from training |

**Outputs (5 variables, PVA protocol):**

| Output PV | Model Variable |
|-----------|---------------|
| `OTRS:IN20:571:XRMS_CU_HXR_LUME` | `OTRS:IN20:571:XRMS` |
| `OTRS:IN20:571:YRMS_CU_HXR_LUME` | `OTRS:IN20:571:YRMS` |
| `OTRS:IN20:571:ZRMS_CU_HXR_LUME` | `sigma_z` |
| `OTRS:IN20:571:EMITN_X_CU_HXR_LUME` | `norm_emit_x` |
| `OTRS:IN20:571:EMITN_Y_CU_HXR_LUME` | `norm_emit_y` |

---

### 2. LCLS FEL Surrogate (`LCLS_FEL_Surrogate`)

| Field | Details |
|-------|---------|
| **Repository** | [LCLS_FEL_Surrogate](https://github.com/SLAClab/LCLS_FEL_Surrogate) |
| **Training Data** | FEL pulse intensity data |
| **Framework** | PyTorch, wrapped with [lume-torch](https://github.com/lume-science/lume-torch) `TorchModel` |
| **Deployment Repo** | [lcls-fel-surrogate-deployment](https://github.com/slaclab/lcls-fel-surrogate-deployment) |
| **Inference Service** | `inference-service-fel` (FastAPI, port 8000) |
| **MLflow Model Name** | `lcls-fel-surrogate` |
| **MLflow Version** | 1 |
| **Device** | CPU |

**Inputs (28 quadrupole magnets, all CA protocol):**

| PV Name | Region |
|---------|--------|
| `QUAD:LI21:211:BACT` | Linac 21 |
| `QUAD:LI21:221:BACT` | Linac 21 |
| `QUAD:LI21:243:BACT` | Linac 21 |
| `QUAD:LI21:251:BACT` | Linac 21 |
| `QUAD:LI21:271:BACT` | Linac 21 |
| `QUAD:LI21:335:BACT` | Linac 21 |
| `QUAD:LI24:713:BACT` | Linac 24 |
| `QUAD:LI24:740:BACT` | Linac 24 |
| `QUAD:LI24:860:BACT` | Linac 24 |
| `QUAD:LI24:892:BACT` | Linac 24 |
| `QUAD:LI24:902:BACT` | Linac 24 |
| `QUAD:CLTH:140:BACT` | CLTH |
| `QUAD:CLTH:170:BACT` | CLTH |
| `QUAD:BSYH:445:BACT` | BSY |
| `QUAD:BSYH:465:BACT` | BSY |
| `QUAD:BSYH:640:BACT` | BSY |
| `QUAD:BSYH:735:BACT` | BSY |
| `QUAD:BSYH:910:BACT` | BSY |
| `QUAD:LTUH:110:BACT` | LTU |
| `QUAD:LTUH:120:BACT` | LTU |
| `QUAD:LTUH:180:BACT` | LTU |
| `QUAD:LTUH:190:BACT` | LTU |
| `QUAD:LTUH:285:BACT` | LTU |
| `QUAD:LTUH:295:BACT` | LTU |
| `QUAD:LTUH:385:BACT` | LTU |
| `QUAD:LTUH:395:BACT` | LTU |
| `QUAD:LTUH:440:BACT` | LTU |
| `QUAD:LTUH:460:BACT` | LTU |
| `QUAD:LTUH:485:BACT` | LTU |
| `QUAD:LTUH:495:BACT` | LTU |
| `QUAD:LTUH:585:BACT` | LTU |
| `QUAD:LTUH:595:BACT` | LTU |
| `QUAD:DMPH:300:BACT` | Dump |
| `QUAD:DMPH:380:BACT` | Dump |
| `QUAD:DMPH:500:BACT` | Dump |
| `QUAD:DMPH:600:BACT` | Dump |

**Model Resources:**
- `lcls_fel_final_model.pt` / `lcls_fel_final_model_cpu.pt` — trained weights
- `lcls_fel_input_scaler.pt` — input normalization
- `lcls_fel_output_scaler.pt` — output denormalization

---

## Deployment Infrastructure

### Repository Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REPOSITORY STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODEL PACKAGES (pip-installable, contain trained weights)                  │
│  ├── lcls_cu_injector_ml_model    → Cu injector NN model                   │
│  └── LCLS_FEL_Surrogate           → FEL pulse intensity model              │
│                                                                             │
│  DEPLOYMENT TEMPLATE (Copier template, generates deployment repos)          │
│  └── lume-model-deployment-template                                         │
│                                                                             │
│  DEPLOYMENT INSTANCES (generated from template, model-specific config)      │
│  ├── lcls-cu-inj-model-deployment → Cu injector online deployment          │
│  └── lcls-fel-surrogate-deployment→ FEL surrogate online deployment        │
│                                                                             │
│  INFERENCE SERVICE (shared FastAPI server, loads models from MLflow)         │
│  └── inference-service            → Generic model serving service           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MLflow Server                                       │
│         https://ard-mlflow.slac.stanford.edu                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐              │
│  │ lcls-cu-inj-model   │  │ lcls-fel-surrogate       │              │
│  │ version: 1          │  │ version: 1               │              │
│  │ artifacts: model.pt │  │ artifacts: model.pt      │              │
│  └─────────┬───────────┘  └──────────────┬───────────┘              │
└────────────┼──────────────────────────────┼──────────────────────────┘
             │ download at startup          │ download at startup
             ▼                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Kubernetes Cluster (S3DF) — namespace: lume-online-ml    │
│                                                                       │
│  ┌─────────────────────────┐    ┌──────────────────────────────┐    │
│  │ inference-service-cu-inj│    │ inference-service-fel         │    │
│  │ (FastAPI :8000)         │    │ (FastAPI :8000)               │    │
│  │ Loads model from MLflow │    │ Loads model from MLflow       │    │
│  │ Serves /predict         │    │ Serves /predict               │    │
│  │ Mem: 4-8Gi, CPU: 1-4   │    │ Mem: 2-4Gi, CPU: 0.5-2       │    │
│  └────────────▲────────────┘    └──────────────▲───────────────┘    │
│               │ HTTP POST                      │ HTTP POST           │
│  ┌────────────┴────────────┐    ┌──────────────┴───────────────┐    │
│  │ lcls-cu-inj-deployment  │    │ lcls-fel-surrogate-deployment│    │
│  │ (Deployment Pod)        │    │ (Deployment Pod)             │    │
│  │ Runs run.py @ 1 Hz     │    │ Runs run.py @ 1 Hz           │    │
│  │ Interface: k2eg/epics   │    │ Interface: k2eg/epics        │    │
│  │ Mem: 2Gi, CPU: 500m    │    │ Mem: 2Gi, CPU: 500m          │    │
│  └────────────┬────────────┘    └──────────────┬───────────────┘    │
└───────────────┼─────────────────────────────────┼────────────────────┘
                │ EPICS CA (read) / PVA (write)   │
                ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LCLS EPICS Control System                          │
│  Input PVs: QUAD:*, SOLN:*, ACCL:*, CAMR:*                          │
│  Output PVs: *_LUME (model predictions published back)               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. MLflow

**Server:** `https://ard-mlflow.slac.stanford.edu/`

**Role:**
- Central model registry — stores versioned model artifacts (PyTorch `.pt` files + config YAML)
- Experiment tracking — logs inference metrics (inputs/outputs) during online operation
- Model metadata — tags for email, repo URL, beam_path, stage

**Workflow:**
1. User registers trained model to MLflow with `mlflow.pytorch.log_model()`
2. Inference service downloads model artifacts at container startup
3. Deployment pods log each inference cycle's inputs/outputs as MLflow metrics

**Model Registration Tags:**
```
email: <contact>
repo: <GitHub URL>
beam_path: cu_hxr / cu_sxr / sc_...
stage: development / production
```

### 2. Inference Service

**Repository:** `inference-service`

**Technology:** FastAPI + Uvicorn, Python 3.10

**Docker Image:** `ghcr.io/slaclab/inference-service/inference-service:latest`

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/model/info` | Model name, version, run_id, input/output names |
| GET | `/inputs` | Input variable specs (names, defaults, ranges, units) |
| GET | `/outputs` | Output variable specs |
| POST | `/predict` | Single prediction `{inputs: dict} → {outputs: dict}` |
| POST | `/predict/batch` | Batch predictions |

**Deployment Pattern:**
- One Docker image, multiple deployments via environment variables (`MODEL_NAME`, `MODEL_VERSION`)
- Each model gets its own K8s Deployment + Service (e.g., `inference-service-cu-inj`, `inference-service-fel`)
- Shared `mlflow-config` ConfigMap provides `MLFLOW_TRACKING_URI`

**K8s Resources (per model):**

| Model | Memory (req/limit) | CPU (req/limit) |
|-------|-------------------|-----------------|
| Cu Injector | 4Gi / 8Gi | 1000m / 4000m |
| FEL Surrogate | 2Gi / 4Gi | 500m / 2000m |

**Health Probes:**
- Liveness: HTTP GET `/health`, initial delay 60s, period 30s
- Readiness: HTTP GET `/health`, initial delay 30s, period 10s

### 3. Deployment Template (`lume-model-deployment-template`)

**Technology:** [Copier](https://copier.readthedocs.io/) with Jinja2 templates

**Purpose:** Generates a complete deployment repository for any LUME model, including:
- `Dockerfile` — Multi-stage build using [Pixi](https://pixi.sh) for environment management
- `deployment.yaml` — Kubernetes manifest
- `src/online_model/` — Inference loop, EPICS interface, MLflow logging, client code

**Key Template Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `registered_model_name` | MLflow model name | — |
| `model_version` | MLflow version | 1 |
| `inference_service_url` | Service endpoint | `http://inference-service:8000` |
| `deployment_name` | K8s deployment name | — |
| `rate` | Inference rate (Hz) | 1 |
| `interface` | `epics`, `k2eg`, or `test` | `k2eg` |
| `device` | `cpu` or `gpu` | `cpu` |

**Generated Inference Loop (`run.py`):**
1. Read input PVs via EPICS CA / k2eg
2. Apply input transformations (sympy formulas from `pv_mapping.yaml`)
3. Call inference service HTTP `/predict` endpoint
4. Apply output transformations
5. Publish predictions to output PVs (PVA protocol)
6. Log inputs/outputs to MLflow
7. Sleep until next cycle (rate-controlled)

### 4. Orchestration & CI/CD

#### GitHub Actions (CI)
- Triggered on push to `main` branch
- Builds Docker image from `Dockerfile`
- Pushes to GitHub Container Registry: `ghcr.io/slaclab/<model-name>:<commit-sha>`

#### ArgoCD (CD)
- Monitors deployment repositories for changes
- Automatically deploys updated images to the S3DF Kubernetes cluster
- Namespace: `lume-online-ml`
- When a new image is pushed (via CI), ArgoCD syncs the deployment

#### Manual Deployment
```bash
# Apply deployment manifest directly
kubectl apply -f deployment.yaml -n lume-online-ml

# Generate a new deployment from template
copier copy --data-file model-configs/<model>.yaml copier-template-k8s deployments/<model-name>
```

---

## Kubernetes Namespace Layout

```
namespace: lume-online-ml
├── Deployments
│   ├── inference-service-cu-inj-deployment    (inference server for Cu inj)
│   ├── inference-service-fel-deployment       (inference server for FEL)
│   ├── lcls-cu-inj-model-deployment           (inference loop pod)
│   └── lcls-fel-surrogate-deployment          (inference loop pod)
├── Services
│   ├── inference-service-cu-inj  → :8000
│   └── inference-service-fel     → :8000
├── ConfigMaps
│   ├── mlflow-config             (MLFLOW_TRACKING_URI)
│   └── k2eg-config-map           (EPICS gateway config, lcls.ini)
└── Secrets
    └── (EPICS_CA_ADDR_LIST)
```

---

## Data Flow (Single Inference Cycle)

```
1. Deployment pod (run.py) reads live PVs
       │
       │  EPICS CA / k2eg
       ▼
2. Apply pv_mapping.yaml transforms (sympy formulas)
       │
       │  e.g. R_DIST = sqrt(XRMS**2 + YRMS**2)
       ▼
3. POST /predict to inference service
       │
       │  HTTP JSON: {"inputs": {"var1": val1, ...}}
       ▼
4. Inference service runs TorchModel.forward()
       │
       │  (includes input/output scalers)
       ▼
5. Returns predictions: {"outputs": {"var1": val1, ...}}
       │
       ▼
6. Apply output transforms, publish to PVA output PVs
       │
       │  e.g. OTRS:IN20:571:XRMS_CU_HXR_LUME
       ▼
7. Log inputs/outputs to MLflow experiment
```

---

## Adding a New Model

1. **Train model** and wrap with `lume-torch` `TorchModel` (produces `.pt` weights + `model_config.yaml`)
2. **Register to MLflow** at `https://ard-mlflow.slac.stanford.edu/` with appropriate tags
3. **Deploy inference service instance:**
   ```bash
   copier copy --data-file model-configs/<model>.yaml copier-template-k8s deployments/<model-name>
   kubectl apply -f deployments/<model-name>/deployment.yaml -n lume-online-ml
   ```
4. **Generate deployment repo:**
   ```bash
   copier copy lume-model-deployment-template <new-deployment-repo>
   ```
5. **Configure `pv_mapping.yaml`** with input/output PV mappings and formulas
6. **Push to GitHub** → CI builds image → ArgoCD deploys automatically

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lume-torch` | ≥ 3.0 | PyTorch model wrapper (`TorchModel`) |
| `mlflow` | 3.4–3.8 | Model registry & experiment tracking |
| `pyepics` | latest | EPICS Channel Access client |
| `k2eg` | ≥ 0.3.2 | k2eg EPICS gateway interface |
| `torch` | ≥ 2.7.1 | PyTorch runtime (CPU wheels) |
| `fastapi` | latest | HTTP API framework (inference service) |
| `uvicorn` | latest | ASGI server |
| `sympy` | latest | Formula evaluation for PV transforms |
