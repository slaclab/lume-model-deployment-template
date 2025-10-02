import k2eg
from k2eg.serialization import Scalar


class K2EGInterface:
    """
    A class to interface with K2EG for reading and writing process variables (PVs).

    Attributes
    ----------
    k2eg_client : k2eg.dml
        The K2EG client used to interact with the K2EG system.
    """

    def __init__(self, environment_id: str = "lcls", app_name: str = "app-three"):
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

    def get_pv(
        self, pv_name: str, timeout: float = 5.0, proto: str = "ca://"
    ) -> Scalar:
        """
        Retrieves the value of a process variable (PV) from K2EG.

        Parameters
        ----------
        pv_name : str
            The name of the process variable to retrieve.
        timeout : float, optional
            The maximum time to wait for the PV value (default is 5.0 seconds).
        proto : str, optional
            The protocol to use for the PV (default is 'ca://', which stands for Channel Access).
            Other options include 'pva://' for Process Variable Access.

        Returns
        -------
        Scalar
            The value of the process variable.
        """
        return self.k2eg_client.get(proto + pv_name, timeout)

    def put_pv(
        self,
        pv_name: str,
        value: float,
        timeout: float = 10.0,
        proto: str = "ca://",
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
            The protocol to use for the PV (default is 'ca://', which stands for Channel Access).
            Other options include 'pva://' for Process Variable Access.
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

        self.k2eg_client.put(proto + pv_name, serialized_value, timeout)

    def get_input_variables(self, input_pvs: list) -> dict:
        """
        Retrieves the input variables from K2EG.

        Parameters
        ----------
        input_pvs : list
            A list of input variable names to retrieve.

        Returns
        -------
        dict
            A dictionary containing the input variable names and their values.
        """
        input_dict = {}
        for var in input_pvs:
            try:
                k2eg_dict = self.get_pv(var)
                input_dict[var] = {
                    "value": k2eg_dict["value"],
                    "posixseconds": k2eg_dict["timeStamp"]["secondsPastEpoch"],
                }
            except Exception as e:
                raise RuntimeError(f"Failed to get PV {var}: {e}")
        return input_dict

    def close(self):
        """
        Closes the K2EG client connection.
        """
        self.k2eg_client.close()
