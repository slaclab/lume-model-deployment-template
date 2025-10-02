import os
import epics


class EPICSInterface:
    """Interface for interacting with EPICS Process Variables (PVs)."""

    def __init__(self, pv_name_list=None):
        """Check environment variables."""
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
        return results
