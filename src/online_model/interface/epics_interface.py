import os
import logging
import sys
import epics

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class EPICSInterface:
    """Interface for interacting with EPICS Process Variables (PVs)."""

    def __init__(self, pv_name_list=None):
        """Instantiate and check environment variables."""
        self.name = "epics"

        if "EPICS_CA_ADDR_LIST" not in os.environ:
            raise EnvironmentError(
                "EPICS_CA_ADDR_LIST environment variable is not set."
            )
        if "EPICS_CA_AUTO_ADDR_LIST" not in os.environ:
            raise EnvironmentError(
                "EPICS_CA_AUTO_ADDR_LIST environment variable is not set."
            )

        self.pv_objects = None
        if pv_name_list is not None:
            self.create_pvs(pv_name_list)

    def create_pvs(self, pv_name_list):
        """
        Create a list of PV objects.

        Parameters
        ----------
        pv_name_list : list of str
            A list of PV names to create.

        Returns
        -------
        list
            A dict of EPICS PV objects.
        """
        self.pv_objects = {name: epics.PV(name) for name in pv_name_list}

    def get_input_variables(self, input_pvs: list) -> dict:
        """
        Retrieve values and timestamps for a list of EPICS input PVs.

        Parameters
        ----------
        input_pvs : list of str
            List of EPICS PV names to retrieve values for.

        Returns
        -------
        dict
            Dictionary mapping PV names to their values and POSIX timestamps, or error info if retrieval fails.
        """
        results = {}
        for pv in input_pvs:
            pv = self.pv_objects[pv]
            try:
                # Wait for the connection to be established
                if pv.wait_for_connection(timeout=5):
                    time_data = pv.get_timevars()

                    # Extract value and timestamp
                    value = pv.get()
                    timestamp = time_data["posixseconds"]

                    results[pv.pvname] = {"value": value, "posixseconds": timestamp}
                else:
                    results[pv.pvname] = {"error": "Connection failed"}
            except Exception as e:
                results[pv.pvname] = {"error": str(e)}
                logger.error(f"Error retrieving PV {pv.pvname}: {e}")
        return results

    def put_output_variables(self, output_dict: dict):
        """
        Write values to EPICS output PVs.

        Parameters
        ----------
        output_dict : dict
            Dictionary mapping PV names to their values to be written.

        Returns
        -------
        None
        """
        # TODO: need to instantiate PV objects first (WIP)
        for pv_name, value in output_dict.items():
            pv = self.pv_objects[pv_name]
            try:
                # Wait for the connection to be established
                if pv.wait_for_connection(timeout=5):
                    pv.put(value)
                else:
                    logger.error(f"Connection failed for PV {pv.pvname}")
            except Exception as e:
                logger.error(f"Error writing to PV {pv.pvname}: {e}")
