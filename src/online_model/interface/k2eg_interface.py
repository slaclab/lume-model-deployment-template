import logging
import time
import sys
import k2eg
from k2eg.serialization import Scalar
from exceptions import OutputWriteFailure
from threading import Lock
from typing import Optional, Dict, Any

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
    _pv_cache : dict
        Cache storing the latest values and timestamps for monitored PVs.
    _cache_lock : Lock
        Thread lock for safe access to the cache.
    _use_monitor : bool
        Flag to determine whether to use monitor (True) or get (False).
    """

    def __init__(
        self, 
        environment_id: str = "lcls", 
        app_name: str = "app-ad-online-ml",
        use_monitor: bool = True,
        input_pvs: Optional[list] = None,
        input_protos: Optional[list] = None
    ):
        """
        Initializes the K2EGInterface with a K2EG client.

        Parameters
        ----------
        environment_id : str
            The environment ID for the K2EG client (e.g., 'lcls').
        app_name : str
            The application name for the K2EG client (e.g., 'app-ad-online-ml').
        use_monitor : bool, optional
            Whether to use monitor mode for PVs (default is True).
        input_pvs : list, optional
            List of input PV names to monitor. If provided, monitors will be set up immediately.
        input_protos : list, optional
            List of protocols corresponding to input_pvs (default is 'ca' for all).
        """
        self.k2eg_client = k2eg.dml(environment_id, app_name)
        self.name = "k2eg"
        self._use_monitor = use_monitor
        self._pv_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()
        self._monitored_pvs = set()
        
        # If input PVs are provided, set up monitors immediately
        if use_monitor and input_pvs:
            self._setup_monitors(input_pvs, input_protos)
            logger.info(f"Initialized monitoring for {len(input_pvs)} PVs")

    def _setup_monitors(self, pv_list: list, protos: Optional[list] = None):
        """
        Set up monitoring for a list of PVs.

        Parameters
        ----------
        pv_list : list
            List of PV names to monitor.
        protos : list, optional
            List of protocols for each PV (default is 'ca' for all).
        """
        if protos is None:
            protos = ["ca"] * len(pv_list)
        elif len(protos) != len(pv_list):
            raise ValueError(
                f"Length of protos list={len(protos)} must match length of pv_list={len(pv_list)}"
            )

        for pv_name, proto in zip(pv_list, protos):
            self._setup_single_monitor(pv_name, proto)

    def _setup_single_monitor(self, pv_name: str, proto: str = "ca", timeout: float = 5.0):
        """
        Set up monitoring for a single PV.
        
        Gets initial value via get(), then sets up monitor for updates.
        This ensures we have a value even for static PVs.

        Parameters
        ----------
        pv_name : str
            The PV name to monitor.
        proto : str
            The protocol to use.
        timeout : float
            Timeout for the monitor setup.
        """
        full_pv_name = f"{proto}://{pv_name}"
        
        if full_pv_name in self._monitored_pvs:
            return  # Already monitoring
        
        def handler(received_pv_name, data):
            """Callback handler for PV updates"""
            with self._cache_lock:
                self._pv_cache[full_pv_name] = {
                    'data': data,
                    'timestamp': time.time(),
                    'pv_timestamp': data.get('timeStamp', {}).get('secondsPastEpoch', None) if isinstance(data, dict) else None
                }
                logger.debug(f"Monitor update for {pv_name}: value={data.get('value') if isinstance(data, dict) else data}")
        
        try:
            # Step 1: Get initial value (critical for static PVs)
            try:
                initial_value = self.k2eg_client.get(full_pv_name, timeout)
                
                with self._cache_lock:
                    self._pv_cache[full_pv_name] = {
                        'data': initial_value,
                        'timestamp': time.time(),
                        'pv_timestamp': initial_value.get('timeStamp', {}).get('secondsPastEpoch', None) if isinstance(initial_value, dict) else None
                    }
                
                logger.debug(f"Got initial value for {pv_name}: {initial_value.get('value') if isinstance(initial_value, dict) else initial_value}")
            
            except Exception as e:
                logger.warning(f"Failed to get initial value for {full_pv_name}: {e}. Will wait for monitor update.")
            
            # Step 2: Set up monitor for future updates
            self.k2eg_client.monitor(full_pv_name, handler, timeout=timeout)
            self._monitored_pvs.add(full_pv_name)
            logger.info(f"Started monitoring {pv_name}")
            
        except Exception as e:
            logger.error(f"Failed to set up monitor for {full_pv_name}: {e}")
            raise

    def get_pv(self, pv_name: str, timeout: float = 5.0, proto: str = "ca") -> Scalar:
        """
        Retrieves the value of a process variable (PV) from K2EG.

        If monitor mode is enabled, returns the cached value. Otherwise, performs a direct get.

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
        full_pv_name = f"{proto}://{pv_name}"
        
        if not self._use_monitor:
            # Use original get method
            return self.k2eg_client.get(full_pv_name, timeout)
        
        # Monitor mode: set up monitor if not already active
        if full_pv_name not in self._monitored_pvs:
            self._setup_single_monitor(pv_name, proto, timeout)
        
        # Wait for cached value with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._cache_lock:
                if full_pv_name in self._pv_cache:
                    return self._pv_cache[full_pv_name]['data']
            time.sleep(0.01)  # 10ms polling interval
        
        raise TimeoutError(
            f"No value received for {pv_name} within {timeout} seconds. "
            f"Monitor may not be receiving updates."
        )

    def get_input_variables(self, input_pvs: list, protos: list[str] = None, retry_delay: float = 0.15) -> dict:
        """
        Retrieves the input variables from K2EG.

        In monitor mode, this returns cached values with consistent timestamps.
        In get mode, uses the original implementation.

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
        def _protos():
            if protos is None:
                return ["ca"] * len(input_pvs)
            if len(protos) != len(input_pvs):
                raise ValueError(
                    f"Length of protos list={len(protos)} must match length of input_pvs list={len(input_pvs)}"
                )
            return protos

        if not self._use_monitor:
            # Use original implementation
            return self._get_input_variables_original(input_pvs, _protos(), retry_delay)
        
        # Monitor mode: ensure all PVs are being monitored
        proto_list = _protos()
        for pv_name, proto in zip(input_pvs, proto_list):
            full_pv_name = f"{proto}://{pv_name}"
            if full_pv_name not in self._monitored_pvs:
                self._setup_single_monitor(pv_name, proto)
        
        # Get cached values
        return self._get_cached_input_variables(input_pvs, proto_list, retry_delay)

    def _get_cached_input_variables(self, input_pvs: list, protos: list, retry_delay: float) -> dict:
        """
        Retrieve input variables from cache with retry logic.
        
        Parameters
        ----------
        input_pvs : list
            List of PV names.
        protos : list
            List of protocols.
        retry_delay : float
            Delay between retries.
            
        Returns
        -------
        dict
            Dictionary of PV values with timestamps.
        """
        m = len(input_pvs)
        attempt = 0
        
        while True:  # Keep retrying forever
            attempt += 1
            result = {}
            
            with self._cache_lock:
                # Check if all PVs have cached values
                all_available = True
                for pv_name, proto in zip(input_pvs, protos):
                    full_pv_name = f"{proto}://{pv_name}"
                    
                    if full_pv_name not in self._pv_cache:
                        all_available = False
                        logger.warning(f"Attempt {attempt}: PV {pv_name} not yet in cache")
                        break
                    
                    cached_data = self._pv_cache[full_pv_name]['data']
                    result[pv_name] = {
                        'value': cached_data.get('value') if isinstance(cached_data, dict) else cached_data,
                        'posixseconds': cached_data.get('timeStamp', {}).get('secondsPastEpoch') if isinstance(cached_data, dict) else None
                    }
            
            if all_available and len(result) == m:
                # Success - got all PVs
                if attempt > 1:
                    logger.info(f"Successfully retrieved all {m} PVs after {attempt} attempts")
                return result
            
            # Failed to get all PVs, log and retry
            logger.warning(f"Attempt {attempt}: only got {len(result)} out of {m} PVs. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

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
            raise ValueError(f"Length of protos ({len(protos)}) must match length of output_dict ({len(output_dict)}).")

        for (var, value), proto in zip(output_dict.items(), protos):
            last_error = None
        
            for attempt in range(max_retries):
                try:
                    self.put_pv(var, value, proto=proto)
                    # Success
                    if attempt > 0:
                        logging.info(f"Successfully put PV {var}")
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
                            f"Outputs are now stale - iteration will restart with fresh inputs."
                        )
                        raise OutputWriteFailure(
                        f"Failed to put PV {var} after {max_retries} attempts. "
                        f"Last error: {last_error}"
                    )

    def close(self):
        """
        Closes the K2EG client connection.
        """
        self.k2eg_client.close()