import argparse
import sys
import logging
import time
import collections
from pathlib import Path
import yaml
import mlflow
from online_model.mlflow_utils import MLflowRun, MLflowModelGetter
from online_model.configs.template_config import (
    registered_model_name,
    rate,
)
from online_model.transformers.transformer import (
    InputPVTransformer,
    OutputPVTransformer,
)
import os


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

PIXI_LOCKFILE_PATH = "/app/pixi.lock"  # Path to the pixi lockfile inside the container
CONFIG_PATH = Path(__file__).parent / "configs" / "pv_mapping.yaml"
model_version = os.environ.get("MODEL_VERSION", None)


class MultiLineDict(collections.UserDict):
    def __str__(self):
        return "\n" + "\n".join(f"{k} = {v}" for k, v in self.data.items())


def get_interface(interface_name, pvname_list=None):
    if interface_name == "test":
        from online_model.interface.test_interface import TestInterface

        return TestInterface()
    elif interface_name == "epics":
        from online_model.interface.epics_interface import EPICSInterface

        return EPICSInterface(pvname_list)
    elif interface_name == "k2eg":
        from online_model.interface.k2eg_interface import K2EGInterface

        return K2EGInterface()
    else:
        raise ValueError(f"Unknown interface: {interface_name}")


def get_model_inputs(model, interface, input_pv_transformer):
    """
    Step 1: Retrieve and transform inputs for the model based on the interface.
    Handles test, epics, and k2eg interfaces, including timestamp extraction and logging.

    Parameters
    ----------
    model : LUMEBaseModel
        The lume-model wrapped model instance to evaluate.
    interface : Interface
         The interface instance (TestInterface, EPICSInterface, or K2EGInterface) for input retrieval.
    input_pv_transformer : InputPVTransformer
        The transformer to map and transform input PVs to model inputs.

    Returns
    -------
    input_dict : dict
        The dictionary of input values for the model.
    max_posixseconds : int or None
        The maximum posixseconds timestamp from inputs, if applicable.
    """
    if interface.name == "test":
        # Get random input variable (within the ranges) from the interface
        input_dict = interface.get_input_variables(model.input_variables)
        max_posixseconds = None

    elif interface.name in ("epics", "k2eg"):
        # Get the values of input variables PVs from the interface
        args = {"input_pvs": input_pv_transformer.input_list}
        if interface.name == "k2eg":
            args["protos"] = input_pv_transformer.proto_list
        input_dict_raw = interface.get_input_variables(**args)

        # Save the latest timestamp from EPICS PVs for logging
        max_posixseconds = int(max(d["posixseconds"] for d in input_dict_raw.values()))
        logger.debug(f"Raw input values from EPICS: {MultiLineDict(input_dict_raw)}")

        # Get model inputs from PV inputs based on formulas defined in pv_mapping.yaml
        input_dict = input_pv_transformer.transform(input_dict_raw)
        logger.debug(
            f"Transformed input values from EPICS: {MultiLineDict(input_dict)}"
        )

    else:
        raise ValueError(f"Unknown interface: {interface.name}")

    logger.debug("Input values: %s", MultiLineDict(input_dict))
    return input_dict, max_posixseconds


def evaluate_model(model, input_dict):
    """
    Step 2: Evaluate the model with the given inputs.
    Sets input validation config and runs model.evaluate.

    Parameters
    ----------
    model : LUMEBaseModel
        The lume-model wrapped model instance to evaluate.
    input_dict : dict
        The dictionary of input values for the model.

    Returns
    -------
    output : dict
        The dictionary of output values from the model.
    """
    # Set custom input validation to warn on all inputs
    # TODO: make this optional? or a config? for now, just warn on all inputs
    model.input_validation_config = {k: "warn" for k in model.input_names}
    output = model.evaluate(input_dict)
    logger.debug("Model output values: %s", MultiLineDict(output))
    return output


def write_output_and_log(
    output, input_dict, max_posixseconds, interface, output_pv_transformer
):
    """
    Step 3: Write output to PVs if applicable and log metrics to MLflow.
    Handles timestamp for epics/k2eg and logs output values.

    Parameters
    ----------
    output : dict
        The dictionary of output values from the model.
    input_dict : dict
        The dictionary of input values for the model.
    max_posixseconds : int or None
        The maximum posixseconds timestamp from inputs, if applicable.
    interface : Interface
        The interface instance (TestInterface, EPICSInterface, or K2EGInterface).
    output_pv_transformer : OutputPVTransformer
        The transformer to map and transform model outputs to output PVs.
    """
    # Clean output: convert torch.Tensor values to Python scalars
    cleaned_output = {}
    for k, v in output.items():
        try:
            import torch

            if isinstance(v, torch.Tensor):
                cleaned_output[k] = v.detach().cpu().numpy()
            else:
                cleaned_output[k] = v
        except ImportError:
            cleaned_output[k] = v

    # Write output to PVs if applicable
    if interface.name in ("epics", "k2eg") and output_pv_transformer is not None:
        output_pv = output_pv_transformer.transform(cleaned_output)
        args = {"output_dict": output_pv}
        if interface.name == "k2eg":
            args["protos"] = output_pv_transformer.proto_list
        interface.put_output_variables(**args)
        logger.debug(
            f"Mapped output values to write to EPICS: {MultiLineDict(output_pv)}"
        )
    elif interface.name == "test":
        logger.info("No PV writing for test interface.")
        pass

    # TODO: add epics timestamp to DB as well, and log all to wall clock time
    # Writing model outputs to MLflow (not outputs mapped to PVs)
    mlflow.log_metrics(
        input_dict | output,
        timestamp=(
            max_posixseconds * 1000
            if max_posixseconds and interface.name in ("epics", "k2eg")
            else None
        ),
    )
    logger.info("Wrote input and output metrics to MLflow.")


def run_iteration(model, interface, input_pv_transformer, output_pv_transformer):
    """
    Orchestrates a single iteration of the model evaluation using the specified interface.
    Step 1: Input retrieval and transformation
    Step 2: Model evaluation
    Step 3: Output writing and logging

    Parameters
    ----------
    model : LUMEBaseModel
        The lume-model wrapped model instance to evaluate.
    interface : Interface
        The interface instance (TestInterface, EPICSInterface, or K2EGInterface) for
        input retrieval.
    input_pv_transformer : InputPVTransformer
        The transformer to map and transform input PVs to model inputs.
    output_pv_transformer : OutputPVTransformer
        The transformer to map and transform model outputs to output PVs.

    Returns
    -------
    None
    """
    input_dict, max_posixseconds = get_model_inputs(
        model, interface, input_pv_transformer
    )
    output = evaluate_model(model, input_dict)
    write_output_and_log(
        output, input_dict, max_posixseconds, interface, output_pv_transformer
    )


def main():
    """
    Main entry point for running the online model application with CLI interface selection.

    Parses command-line arguments to select the interface, initializes the model and interface,
    and runs the evaluation loop.

    You can run the script with:
        python run.py --interface test
        python run.py --interface epics

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Run the model with selected interface."
    )
    parser.add_argument(
        "--interface",
        "-i",
        choices=["test", "epics", "k2eg"],
        required=True,
        help="Interface to use",
    )
    args = parser.parse_args()
    logger.info("Running with interface: %s", args.interface)

    # Load the model from MLflow Model Registry
    model = MLflowModelGetter(registered_model_name, model_version).get_model()

    # Set up PV transformer
    # This is required to map from EPICS PV names to model input names, and apply any formulas
    # defined in configs/pv_mapping.yaml. This is applicable only for EPICS/k2eg interfaces, and is in addition
    # to the lume-model's own internal input_transform method, if any are defined.
    with open(CONFIG_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)
    input_pv_transformer = InputPVTransformer(config_yaml)
    if "output_variables" in config_yaml:
        # User defined output variable mapping
        output_pv_transformer = OutputPVTransformer(config_yaml)
    else:
        output_pv_transformer = None

    interface = get_interface(
        args.interface,
        input_pv_transformer.input_list if args.interface == "epics" else None,
    )

    run_tags = {
        "interface": args.interface,
        "model_name": registered_model_name,
        "model_version": model_version,
    }

    with MLflowRun(tags=run_tags) as _:
        # Log lockfile for complete reproducibility
        try:
            mlflow.log_artifact(PIXI_LOCKFILE_PATH, "pixi_lockfile")
        except FileNotFoundError:
            logger.error(
                f"Lockfile {PIXI_LOCKFILE_PATH} not found. Continuing without logging it."
            )
        # Run the evaluation loop
        while True:
            try:
                run_iteration(
                    model, interface, input_pv_transformer, output_pv_transformer
                )
                time.sleep(rate)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Exiting.")
                break
            except Exception as e:
                raise e


if __name__ == "__main__":
    main()
