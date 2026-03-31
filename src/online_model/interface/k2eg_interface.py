import logging
import time
import sys
import threading
from typing import Dict, Any
import k2eg
from k2eg.serialization import Scalar
from exceptions import OutputWriteFailure

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
    Uses monitors for efficient real-time data acquisition.

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
            The application name for the K2EG client (e.g., 'app-ad-online-ml').
        """
        self.k2eg_client = k2eg.dml(environment_id, app_name)
        self.name = "k2eg"
        
        # Monitor state management
        self._pv_data: Dict[str, Dict[str, Any]] = {}  # Stores latest PV values
        self._pv_ready: Dict[str, threading.Event] = {}  # Signals when PV has data
        self._pv_locks: Dict[str, threading.Lock] = {}  # Thread-safe access
        self._monitors_started = False
        self._monitor_setup_lock = threading.Lock()  # Ensure setup happens once

    def _monitor_handler(self, pv_name: str, data: Any):
        """
        Handler called by k2eg monitor when PV updates.
        Stores the latest value and signals that the PV is ready.

        Parameters
        ----------
        pv_name : str
            The name of the PV that was updated.
        data : Any
            The updated data from k2eg.
        """
        with self._pv_locks[pv_name]:
            self._pv_data[pv_name] = data
            self._pv_ready[pv_name].set()  # Signal that this PV has data
            logger.debug(f"Monitor update for {pv_name}: value={data.get('value', 'N/A')}")

    def _setup_monitors_if_needed(self, input_pvs: list, protos: list[str], timeout: float = 5.0):
        """
        Set up monitors for all input PVs if not already started.
        Thread-safe - will only set up once even if called multiple times.

        Parameters
        ----------
        input_pvs : list
            A list of input variable names to monitor.
        protos : list of str
            A list of protocols corresponding to each input variable.
        timeout : float, optional
            Timeout for establishing monitors (default is 5.0 seconds).
        """
        with self._monitor_setup_lock:
            if self._monitors_started:
                return  # Already started, nothing to do
            
            logger.info(f"Setting up monitors for {len(input_pvs)} PVs...")
            
            for pv_name, proto in zip(input_pvs, protos):
                # Initialize data structures for this PV
                self._pv_locks[pv_name] = threading.Lock()
                self._pv_ready[pv_name] = threading.Event()
                
                # Create a handler for this specific PV using a closure
                def create_handler(pv):
                    return lambda data: self._monitor_handler(pv, data)
                
                handler = create_handler(pv_name)
                
                # Start monitoring
                full_pv_name = f"{proto}://{pv_name}"
                try:
                    self.k2eg_client.monitor(full_pv_name, handler, timeout=timeout)
                    logger.info(f"Started monitor for {full_pv_name}")
                except Exception as e:
                    logger.error(f"Failed to start monitor for {full_pv_name}: {e}")
                    raise
            
            self._monitors_started = True
            logger.info("All monitors started successfully")
            
            # Wait for monitors to get initial data
            logger.info("Waiting for monitors to receive initial data...")
            time.sleep(2.0)
            logger.info("Monitors ready")

    def get_pv(self, pv_name: str, timeout: float = 5.0, proto: str = "ca") -> Scalar:
        """
        Retrieves the value of a process variable (PV) from K2EG.
        
        NOTE: This method is kept for backward compatibility but should not be used
        when monitors are set up. Use get_input_variables() instead.

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
        value : float
            The value to write to the process variable.
        timeout : float, optional
            The maximum time to wait for the write operation (default is 10.0 seconds).
        proto : str, optional
            The protocol to use for the PV (default is 'ca', which stands for Channel Access).
            Other options include 'pva' for Process Variable Access.
        type : str, optional
            The type of value (default is 'scalar').
        """
        if type == "scalar":
            if not isinstance(value, float):
                raise TypeError("Value must be a float.")
            serialized_value = Scalar("value", value)
        else:
            # Dict, lists and NTTable are supported in k2eg, but not implemented here.
            raise NotImplementedError(
                f"Unsupported type: {type}. Only 'scalar' is supported."
            )

        self.k2eg_client.put(proto + "://" + pv_name, serialized_value, timeout)

    def get_input_variables(self, input_pvs: list, protos: list[str] = None, retry_delay: float = 0.15) -> dict:
        """
        Retrieves the input variables from K2EG monitors.
        Automatically sets up monitors on first call.
        Waits until all PVs have received at least one update, then reads
        them all quickly to get consistent timestamps.

        Parameters
        ----------
        input_pvs : list
            A list of input variable names to retrieve.
        protos : list of str, optional
            A list of protocols corresponding to each input variable (default is 'ca' for all).
        retry_delay : float, optional
            Delay in seconds between retries (default is 0.15, 150ms between retries).

        Returns
        -------
        dict
            A dictionary containing the input variable names and their values.
        """

        def _all_pvs(input_pvs):
            """Get PVs with consistent timestamps from monitor cache"""
            for var in input_pvs:
                with self._pv_locks[var]:
                    rv = self._pv_data.get(var)
                    if rv is None:
                        return  # PV not ready yet
                    yield var, dict(
                        value=rv["value"], 
                        posixseconds=rv["timeStamp"]["secondsPastEpoch"]
                    )

        def _protos():
            if protos is None:
                return ["ca"] * len(input_pvs)
            if len(protos) != len(input_pvs):
                raise ValueError(
                    f"Length of protos list={len(protos)} must match length of input_pvs list={len(input_pvs)}"
                )
            return protos

        def _try_pvs(input_pvs):
            m = len(input_pvs)
            attempt = 0
            
            while True:  # Keep retrying forever
                attempt += 1
                rv = tuple(_all_pvs(input_pvs))
                
                if len(rv) == m:
                    # Success
                    if attempt > 1:
                        logger.info(f"Successfully retrieved all {m} input variables on attempt {attempt}")
                    return rv
                
                # Failed to get all PVs, log and retry
                not_ready = [pv for pv in input_pvs if not self._pv_ready[pv].is_set()]
                msg = f"only got len(pvs)={len(rv)} out of expected={m}"
                
                if attempt % 10 == 0:  # Log every 10th attempt
                    logger.warning(
                        f"Attempt {attempt}: {msg}. "
                        f"Not ready: {not_ready[:5]}{'...' if len(not_ready) > 5 else ''}. Still retrying..."
                    )
                elif attempt <= 5:  # Log first 5 attempts
                    logger.warning(f"Attempt {attempt}: {msg}. Retrying in {retry_delay}s...")
                
                time.sleep(retry_delay)
        
        # Setup monitors automatically on first call
        validated_protos = _protos()
        self._setup_monitors_if_needed(input_pvs, validated_protos)
        
        return dict(_try_pvs(input_pvs))

    def put_output_variables(self, output_dict: dict, protos: list = None, max_retries: int = 3, retry_delay: float = 0.1):
        """
        Writes the output variables to K2EG.
        
        If any PV fails after max_retries, raises OutputWriteFailure to signal
        that the current iteration should be abandoned and restarted with fresh inputs.

        Parameters
        ----------
        output_dict: dict
            A dictionary containing the output variable names and their values.
        protos: list of str, optional
            A list of protocols corresponding to each output variable (default is 'ca' for all).
        max_retries : int, optional
            Maximum number of retry attempts per PV (default is 3).
        retry_delay : float, optional
            Delay in seconds between retries (default is 0.1, 100ms between retries).

        Returns
        -------
        None
        
        Raises
        ------
        OutputWriteFailure
            If writing any PV fails after max_retries.
        """
        if protos is None:
            protos = ["ca"] * len(output_dict)
        elif len(protos) != len(output_dict):
            raise ValueError(f"Length of protos ({len(protos)}) must match length of output_dict ({len(output_dict)}).")

        for (var, value), proto in zip(output_dict.items(), protos):
            last_error = None
        
            for attempt in range(max_retries):
                try:
                    self.put_pv(var, value, proto=proto)
                    # Success
                    if attempt > 0:
                        logger.info(f"Successfully put PV {var} on attempt {attempt + 1}")
                    break  # Move to next PV
                
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Transient failure putting PV {var} (attempt {attempt + 1}/{max_retries}): {e}. Retrying..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to put PV {var} after {max_retries} attempts: {e}. "
                            f"Outputs are now stale - iteration will restart with fresh inputs."
                        )
                        raise OutputWriteFailure(
                            f"Failed to put PV {var} after {max_retries} attempts. "
                            f"Last error: {last_error}"
                        )

    def close(self):
        """
        Closes the K2EG client connection and cleans up monitors.
        """
        logger.info("Closing K2EG interface and cleaning up monitors...")
        self._monitors_started = False
        # Clear monitor data
        self._pv_data.clear()
        self._pv_ready.clear()
        self._pv_locks.clear()
        # Close k2eg client
        self.k2eg_client.close()