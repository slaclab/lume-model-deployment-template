import logging
import time
import sys
import threading
from typing import Dict, Any, Optional
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
        self._pv_locks: Dict[str, threading.Lock] = {}  # Thread-safe access per PV
        self._data_lock = threading.Lock()  # Global lock for _pv_data access
        self._any_pv_updated = threading.Event()  # Signals when at least one PV has data
        self._expected_pvs: set = set()  # Track which PVs we're monitoring
        self._monitors_started = False
        self._monitor_setup_lock = threading.Lock()  # Ensure setup happens once
        self._monitor_errors: Dict[str, str] = {}  # Track monitor setup errors

    def _monitor_handler(self, pv_name: str, data: Any):
        """
        Handler called by k2eg monitor when PV updates.
        Stores the latest value and signals that at least one PV is ready.

        Parameters
        ----------
        pv_name : str
            The name of the PV that was updated.
        data : Any
            The updated data from k2eg.
        """
        try:
            with self._data_lock:
                self._pv_data[pv_name] = data
                received_count = len(self._pv_data)
                expected_count = len(self._expected_pvs)
                
                # Signal that at least one PV has received data
                self._any_pv_updated.set()
                
                logger.info(
                    f"✓ Monitor update for {pv_name}: value={data.get('value', 'N/A')} "
                    f"({received_count}/{expected_count} PVs ready)"
                )
        except Exception as e:
            logger.error(f"Error in monitor handler for {pv_name}: {e}", exc_info=True)

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
            self._expected_pvs = set(input_pvs)
            
            successful_monitors = 0
            for pv_name, proto in zip(input_pvs, protos):
                # Initialize lock for this PV
                self._pv_locks[pv_name] = threading.Lock()
                
                # Create a handler for this specific PV using a closure
                def create_handler(pv):
                    return lambda data: self._monitor_handler(pv, data)
                
                handler = create_handler(pv_name)
                
                # Start monitoring
                full_pv_name = f"{proto}://{pv_name}"
                try:
                    logger.debug(f"Attempting to start monitor for {full_pv_name}...")
                    self.k2eg_client.monitor(full_pv_name, handler, timeout=timeout)
                    logger.info(f"✓ Successfully started monitor for {full_pv_name}")
                    successful_monitors += 1
                except Exception as e:
                    error_msg = f"Failed to start monitor: {e}"
                    self._monitor_errors[pv_name] = error_msg
                    logger.error(f"✗ Monitor setup failed for {full_pv_name}: {e}")
                    # Don't raise - continue trying other PVs
            
            if successful_monitors == 0:
                error_summary = "\n".join([f"  {pv}: {err}" for pv, err in self._monitor_errors.items()])
                raise RuntimeError(
                    f"Failed to start any monitors (0/{len(input_pvs)}). Errors:\n{error_summary}"
                )
            
            self._monitors_started = True
            logger.info(
                f"Monitor setup complete: {successful_monitors}/{len(input_pvs)} monitors started successfully"
            )
            
            if self._monitor_errors:
                logger.warning(
                    f"Some monitors failed to start: {list(self._monitor_errors.keys())}"
                )

    def _wait_for_initial_data(self, timeout: float = 30.0) -> bool:
        """
        Wait for at least one PV to receive data.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for first PV update (default is 30.0 seconds).
            
        Returns
        -------
        bool
            True if at least one PV received data, False if timeout occurred.
        """
        logger.info(f"Waiting for at least one PV to receive data (timeout: {timeout}s)...")
        
        # Check every second and log progress
        start_time = time.time()
        check_interval = 1.0
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            if self._any_pv_updated.wait(timeout=check_interval):
                with self._data_lock:
                    received = len(self._pv_data)
                    expected = len(self._expected_pvs)
                    missing = self._expected_pvs - set(self._pv_data.keys())
                    
                    logger.info(f"✓ At least one PV ready. Status: {received}/{expected} PVs have data")
                    if missing and received < expected:
                        missing_list = list(missing)[:5]
                        logger.warning(
                            f"Missing PVs: {missing_list}"
                            f"{'...' if len(missing) > 5 else ''} "
                            f"({len(missing)} total)"
                        )
                    return True
            
            # Log progress every 5 seconds
            if time.time() - last_log_time >= 5.0:
                with self._data_lock:
                    received = len(self._pv_data)
                elapsed = time.time() - start_time
                logger.info(
                    f"Still waiting... ({elapsed:.1f}s elapsed, {received} PVs received so far)"
                )
                last_log_time = time.time()
        
        # Timeout occurred
        logger.error(f"✗ Timeout: No PV updates received in {timeout}s")
        
        # Provide diagnostic information
        with self._data_lock:
            if self._pv_data:
                logger.info(f"Received data from: {list(self._pv_data.keys())}")
            else:
                logger.error("No data received from any PV")
        
        if self._monitor_errors:
            logger.error(f"Monitor setup errors occurred for: {list(self._monitor_errors.keys())}")
        
        return False

    def get_missing_pvs(self) -> list:
        """
        Get list of PVs that haven't received updates yet.
        
        Returns
        -------
        list
            List of PV names that haven't received data.
        """
        with self._data_lock:
            return list(self._expected_pvs - set(self._pv_data.keys()))

    def get_monitor_status(self) -> dict:
        """
        Get detailed status of all monitors.
        
        Returns
        -------
        dict
            Dictionary with status information:
            - 'total': total number of expected PVs
            - 'ready': number of PVs with data
            - 'missing': list of PVs without data
            - 'errors': dict of PVs with setup errors
        """
        with self._data_lock:
            return {
                'total': len(self._expected_pvs),
                'ready': len(self._pv_data),
                'missing': self.get_missing_pvs(),
                'errors': self._monitor_errors.copy()
            }

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

    def get_input_variables(
        self, 
        input_pvs: list, 
        protos: list[str] = None, 
        initial_timeout: float = 30.0,
        fallback_to_get: bool = True
    ) -> dict:
        """
        Retrieves the input variables from K2EG monitors.
        Automatically sets up monitors on first call.
        
        NEW BEHAVIOR: Proceeds as soon as at least ONE PV receives data, rather than
        waiting for all PVs. This prevents blocking when some PVs are offline or slow.

        Parameters
        ----------
        input_pvs : list
            A list of input variable names to retrieve.
        protos : list of str, optional
            A list of protocols corresponding to each input variable (default is 'ca' for all).
        initial_timeout : float, optional
            Maximum time to wait for first PV update on initial setup (default is 30.0 seconds).
            Only used on first call when monitors are being set up.
        fallback_to_get : bool, optional
            If True and monitors fail, fall back to using get_pv() for each PV (default is True).

        Returns
        -------
        dict
            A dictionary containing the input variable names and their values.
            Only includes PVs that have received data.
        """

        def _get_available_pvs(input_pvs):
            """Get all PVs that currently have data"""
            with self._data_lock:
                result = {}
                for var in input_pvs:
                    rv = self._pv_data.get(var)
                    if rv is not None:
                        result[var] = dict(
                            value=rv["value"], 
                            posixseconds=rv["timeStamp"]["secondsPastEpoch"]
                        )
                return result

        def _protos():
            if protos is None:
                return ["ca"] * len(input_pvs)
            if len(protos) != len(input_pvs):
                raise ValueError(
                    f"Length of protos list={len(protos)} must match length of input_pvs list={len(input_pvs)}"
                )
            return protos
        
        def _fallback_get_pvs(input_pvs, protocols):
            """Fallback to using get_pv() if monitors fail"""
            logger.warning("Falling back to get_pv() method for reading PVs")
            result = {}
            for pv_name, proto in zip(input_pvs, protocols):
                try:
                    value = self.get_pv(pv_name, proto=proto)
                    result[pv_name] = dict(
                        value=value.value,
                        posixseconds=time.time()  # Use current time
                    )
                    logger.debug(f"Got value for {pv_name} via get_pv(): {value.value}")
                except Exception as e:
                    logger.warning(f"Failed to get {pv_name} via get_pv(): {e}")
            return result
        
        # Setup monitors automatically on first call
        validated_protos = _protos()
        
        # First time setup
        if not self._monitors_started:
            try:
                self._setup_monitors_if_needed(input_pvs, validated_protos)
                
                # Wait for at least one PV to get data
                if not self._wait_for_initial_data(timeout=initial_timeout):
                    logger.error("Failed to receive any PV updates during initial setup")
                    
                    # Show diagnostic info
                    status = self.get_monitor_status()
                    logger.error(
                        f"Monitor status: {status['ready']}/{status['total']} ready. "
                        f"Errors: {len(status['errors'])}"
                    )
                    
                    # Try fallback method if enabled
                    if fallback_to_get:
                        logger.info("Attempting fallback to get_pv() method...")
                        return _fallback_get_pvs(input_pvs, validated_protos)
                    else:
                        return {}  # Return empty dict if no fallback
                        
            except Exception as e:
                logger.error(f"Error setting up monitors: {e}", exc_info=True)
                
                # Try fallback method if enabled
                if fallback_to_get:
                    logger.info("Attempting fallback to get_pv() method...")
                    return _fallback_get_pvs(input_pvs, validated_protos)
                else:
                    raise
        
        # Get all currently available PV data from monitors
        available_data = _get_available_pvs(input_pvs)
        
        # If no data available from monitors and fallback is enabled, try get_pv
        if not available_data and fallback_to_get:
            logger.warning("No monitor data available, using fallback get_pv() method")
            return _fallback_get_pvs(input_pvs, validated_protos)
        
        # Log status if not all PVs are available
        if len(available_data) < len(input_pvs):
            missing = self.get_missing_pvs()
            logger.debug(
                f"Returning {len(available_data)}/{len(input_pvs)} PVs. "
                f"Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
            )
        
        return available_data

    def put_output_variables(
        self, 
        output_dict: dict, 
        protos: list = None, 
        max_retries: int = 3, 
        retry_delay: float = 0.1
    ):
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
            raise ValueError(
                f"Length of protos ({len(protos)}) must match length of output_dict ({len(output_dict)})."
            )

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
        with self._monitor_setup_lock:
            self._monitors_started = False
        
        # Clear monitor data
        with self._data_lock:
            self._pv_data.clear()
            self._expected_pvs.clear()
        
        self._pv_locks.clear()
        self._monitor_errors.clear()
        self._any_pv_updated.clear()
        
        # Close k2eg client
        try:
            self.k2eg_client.close()
            logger.info("K2EG client closed successfully")
        except Exception as e:
            logger.error(f"Error closing k2eg client: {e}")