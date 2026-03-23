import logging
import time
import sys
import k2eg
from k2eg.serialization import Scalar

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class K2EGInterface:
    """
    A class to interface with K2EG for reading and writing process variables (PVs).

    Attributes
    ----------
    k2eg_client : k2eg.dml
        The K2EG client used to interact with the K2EG system.
    """

    def __init__(
        self, environment_id: str = "lcls", app_name: str = "app-ad-online-ml"
    ):
        """
        Initializes the K2EGInterface with a K2EG client.

        Parameters
        ----------
        environment_id : str
            The environment ID for the K2EG client (e.g., 'lcls').
        app_name : str
            The application name for the K2EG client (e.g., 'app-three').
        """
        self.k2eg_client = k2eg.dml(environment_id, app_name)
        self.name = "k2eg"

    def get_pv(self, pv_name: str, timeout: float = 5.0, proto: str = "ca") -> Scalar:
        """
        Retrieves the value of a process variable (PV) from K2EG.

        Parameters
        ----------
        pv_name : str
            The name of the process variable to retrieve.
        timeout : float, optional
            The maximum time to wait for the PV value (default is 5.0 seconds).
        proto : str, optional
            The protocol to use for the PV (default is 'ca', which stands for Channel Access).
            Other options include 'pva' for Process Variable Access.

        Returns
        -------
        Scalar
            The value of the process variable.
        """
        return self.k2eg_client.get(proto + "://" + pv_name, timeout)

    def put_pv(
        self,
        pv_name: str,
        value: float,
        timeout: float = 10.0,
        proto: str = "ca",
        type: str = "scalar",
    ):
        """
        Writes a value to a process variable (PV) in K2EG.

        Parameters
        ----------
        pv_name : str
            The name of the process variable to write to.
        value : Scalar
            The value to write to the process variable.
        timeout : float, optional
            The maximum time to wait for the write operation (default is 10.0 seconds).
        proto : str, optional
            The protocol to use for the PV (default is 'ca', which stands for Channel Access).
            Other options include 'pva' for Process Variable Access.
        """
        if type == "scalar":
            if not isinstance(value, float):
                raise TypeError("Value must be an instance of Scalar.")
            serialized_value = Scalar("value", value)
        else:
            # Dict, lists and NTTable are supported in k2eg, but not implemented here.
            raise NotImplementedError(
                f"Unsupported type: {type}. Only 'scalar' is supported."
            )

        self.k2eg_client.put(proto + "://" + pv_name, serialized_value, timeout)

    def get_input_variables(self, input_pvs: list, protos: list[str] = None, max_retries: int = 3, retry_delay: float = 0.15) -> dict:
        """
        Retrieves the input variables from K2EG.

        Parameters
        ----------
        input_pvs : list
            A list of input variable names to retrieve.
        protos : list of str, optional
            A list of protocols corresponding to each input variable (default is 'ca' for all).
        max_retries : int, optional
            Maximum number of retry attempts (default is 3).
        retry_delay : float, optional
            Delay in seconds between retries (default is 0.15, 150ms between retries).

        Returns
        -------
        dict
            A dictionary containing the input variable names and their values.
        """
        

        def _all_pvs(protos):
            """get pvs with consistent timestamps"""
            for var, proto in zip(input_pvs, protos):
                if (rv := _pv(var, proto)) is None:
                    return
                yield var, dict(value=rv["value"], posixseconds=rv["timeStamp"]["secondsPastEpoch"])

        def _protos():
            if protos is None:
                return ["ca"] * len(input_pvs)
            if len(protos) != len(input_pvs):
                raise ValueError(
                    f"Length of protos list={len(protos)} must match length of input_pvs list={len(input_pvs)}"
                )
            return protos

        def _pv(var, proto):
            try:
                return self.get_pv(var, proto=proto)
            except Exception as e:
                return None
        
        def _try_pvs(protos):
            m = len(input_pvs)
            for _ in range(max_retries):
                rv = tuple(_all_pvs(protos))
                if len(rv) == m:
                    return rv
                e = f"only got len(pvs)={len(rv)} out of expect={m}"
                logging.warning(e)
                time.sleep(retry_delay)
            raise RuntimeError(e)
        
        return dict(_try_pvs(_protos()))

    def put_output_variables(self, output_dict: dict, protos: list = None, max_retries: int = 2, retry_delay: float = 0.1):
        """
        Writes the output variables to K2EG.

        Parameters
        ----------
        output_dict: dict
            A dictionary containing the output variable names and their values.
        protos: list of str, optional
            A list of protocols corresponding to each output variable (default is 'ca' for all).
        max_retries : int, optional
            Maximum number of retry attempts per PV (default is 2).
        retry_delay : float, optional
            Delay in seconds between retries (default is 0.1, 100ms between retries).

        Returns
        -------
        None
        """
        if protos is None:
            protos = ["ca"] * len(output_dict)
        elif len(protos) != len(output_dict):
            raise ValueError("Length of protos list must match length of output_dict.")

        for (var, value), proto in zip(output_dict.items(), protos):
            last_error = None
        
            for attempt in range(max_retries):
                try:
                    self.put_pv(var, value, proto=proto)
                    # Success
                    if attempt > 0:
                        logging.info(f"Successfully put PV {var} on attempt {attempt + 1}")
                    break  # Move to next PV
                
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Transient failure putting PV {var} (attempt {attempt + 1}/{max_retries}): {e}. Retrying..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logging.error(
                            f"Failed to put PV {var} after {max_retries} attempts: {e}"
                        )
                        raise RuntimeError(
                            f"Failed to put PV {var} after {max_retries} attempts. Last error: {last_error}"
                        )

    def close(self):
        """
        Closes the K2EG client connection.
        """
        self.k2eg_client.close()
