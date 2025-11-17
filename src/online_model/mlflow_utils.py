import logging
import mlflow
from mlflow import MlflowClient
from mlflow.models.model import get_model_info
from lume_model.models import TorchModel, TorchModule
from online_model.configs.template_config import mlflow_tracking_uri, deployment_name


logger = logging.getLogger(__name__)


class MLflowRun:
    """
    Context manager for MLflow runs.

    Automatically sets the tracking URI and experiment name, and generates a unique run name
    based on previous runs. Ensures that MLflow runs are properly started and ended.

    Parameters
    ----------
    tracking_uri : str, optional
        The MLflow tracking server URI.
    experiment_name : str, optional
        The name of the MLflow experiment.
    run_prefix : str, optional
        Prefix for run names to distinguish runs.
    """

    def __init__(
        self,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=deployment_name,
        run_prefix=f"{deployment_name} run",
        tags=None,
    ):
        self.run_prefix = run_prefix
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.tags = tags
        self.run_name = self.setup_experiment()

    def __enter__(self):
        """
        Start an MLflow run and return the run object.
        """
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        logger.info(f"Started MLflow run: {self.run_name}")
        return self.run

    def __exit__(self, exc_type, exc_value, traceback):
        """
        End the MLflow run when exiting the context.
        """
        mlflow.end_run()

    def setup_experiment(self):
        """
        Set up the MLflow experiment and generate a unique run name.

        Returns
        -------
        str
            The generated run name for the new MLflow run.
        """
        logger.debug("Setting up MLflow experiment...")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)

        # Get next run name
        all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_numbers = []

        for run in all_runs:
            tag = run.data.tags.get("mlflow.runName", "")
            if tag.startswith(self.run_prefix):
                try:
                    num = int(tag.replace(self.run_prefix, "").strip())
                    run_numbers.append(num)
                except ValueError:
                    continue

        next_run_number = max(run_numbers, default=0) + 1
        return self.run_prefix + str(next_run_number)


class MLflowModelGetter:
    # Adapted from Mat Leputa's poly-lithic implementation
    # https://github.com/ISISNeutronMuon/poly-lithic/blob/main/poly_lithic/src/model_utils/MlflowModelGetter.py
    def __init__(self, model_name, model_version=None, model_uri=None):
        # either supply version or URI
        if "model_version" != None:
            model_uri = None
        elif "model_uri" != None:
            model_version = None
        else:
            raise ValueError("Either model_version or model_uri must be supplied")
        logger.debug(f"MLflowModelGetter: {model_name}, {model_version}, {model_uri}")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.debug((f"MLflow tracking URI: {mlflow.get_tracking_uri()}"))

        self.model_name = model_name
        self.model_version = model_version
        self.model_uri = model_uri
        self.client = MlflowClient()
        self.model_type = None
        self.tags = None

    # def get_requirements(self):
    #     # Determine model version
    #     if isinstance(self.model_version, int) or (
    #             isinstance(self.model_version, str) and self.model_version.isdigit()):
    #         version = self.client.get_model_version(self.model_name, str(self.model_version))
    #     elif self.model_version == "champion":
    #         version_no = self.client.get_model_version_by_alias(self.model_name, self.model_version)
    #         version = self.client.get_model_version(self.model_name, version_no.version)
    #     else:
    #         raise ValueError(
    #             f"Invalid model version: {self.model_version}. Must be a non-negative integer or 'champion'."
    #         )
    #     if not version:
    #         raise ValueError(
    #             f"Model version {self.model_version} not found for model {self.model_name}."
    #         )
    #     # Download requirements.txt and read contents
    #     req_path = mlflow.artifacts.download_artifacts(f"{version.source}/requirements.txt")
    #     with open(req_path, "r") as f:
    #         deps = f.read()
    #     os.remove(req_path)
    #     return deps

    def get_model(self):
        if self.model_uri is not None:
            model_uri = self.model_uri
        elif self.model_version is not None:  # TODO: why is this different from above?
            version = self.client.get_model_version(self.model_name, self.model_version)
            model_uri = version.source
        else:
            raise Exception(
                "Either model_version and model name or model_uri must be supplied"
            )

        # flavor
        flavor = get_model_info(model_uri=model_uri).flavors
        loader_module = flavor["python_function"]["loader_module"]
        logger.debug(f"Loader module: {loader_module}")

        if loader_module == "mlflow.pyfunc.model":
            logger.debug("Loading pyfunc model")
            model_pyfunc = mlflow.pyfunc.load_model(model_uri=model_uri)

            # check if model has.get_lume_model() method
            if not hasattr(model_pyfunc.unwrap_python_model(), "get_lume_model"):
                # check if it has get__model() method
                if not hasattr(model_pyfunc.unwrap_python_model(), "get_model"):
                    raise Exception(
                        "Model does not have get_lume_model() or get_model() method"
                    )
                else:
                    logger.debug("Model has get_model() method")
                    logger.warning(
                        "get_model() suggests a non-LUME model, please check if model has an evaluate method"
                    )
                    model = model_pyfunc.unwrap_python_model().get_model()
            else:
                logger.debug("Model has get_lume_model() method")
                model = model_pyfunc.unwrap_python_model().get_lume_model()

            logger.debug(f"Model: {model}, Model type: {type(model)}")
            self.model_type = "pyfunc"
            return model

        elif loader_module == "mlflow.pytorch":
            print("Loading torch model")
            model_torch_module = mlflow.pytorch.load_model(model_uri=model_uri)
            assert isinstance(model_torch_module, TorchModule)
            model = model_torch_module.model
            assert isinstance(model, TorchModel)
            logger.debug(f"Model: {model}, Model type: {type(model)}")
            self.model_type = "torch"
            return model
        else:
            raise Exception(f"Flavor {flavor} not supported")
