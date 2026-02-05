import requests
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceClient:
    """
    Client class for calling the LUME model inference service.
    This provides methods for interacting with the remote inference service.

    Parameters
    ----------
    base_url: str
        Base url of the inference service (e.g, "https://inference-service:8000")
    timeout: int, optional
        Request timeout in seconds. Default is 30
    
    """
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Args:
            base_url: Base URL of the inference service (e.g., "http://inference-service:8000")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()  # Reuse connection
        logger.info(f"Initialized client for {self.base_url}")
    
    def health_check(self) -> bool:
        """
        Check if service is healthy
        
        Returns
        -------
        bool
            True if the service responds with status 200, False otherwise
        
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata
        
        Returns
        -------
        dict: Dictionary containing model information with keys:
        
        """
        response = self.session.get(
            f"{self.base_url}/model/info",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_inputs(self) -> Dict[str, Any]:
        """
        Get model input specifications
        Returns
        -------
        dict: Dictionary containing input_variables and detailed specs for each input. 
        
        """
        response = self.session.get(
            f"{self.base_url}/inputs",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_outputs(self) -> Dict[str, Any]:
        """
        Get model output specifications
        Returns
        -------
        dict: Dictionary containing output variables and specs (unit).

        """
        response = self.session.get(
            f"{self.base_url}/outputs",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a single prediction using inference service.
        Supports partial inputs - the model will use default values for any
        missing input variables
        
        Parameters
        ----------
        inputs: dict of {str: float}
                Dictionary mapping input variable names to their values.
            
        Returns
        -------
        outputs: dict of {str: float} 
                 Dictionary with prediction results mapping output variable names
                to their predicted values. 
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json={"inputs": inputs},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, inputs_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Make batch predictions using the inference service.
        Sends multiple input samples in a single request for 
        batch processing. Each input dictionary can have partial inputs.
        
        Parameters
        ----------
        inputs_list: list of dict
                      List of input dictionaries (each can be partial), 
                      where each dictionary maps input variable names to their values.
            
        Returns
        -------
        dict:  Dictionary containing:
                - 'outputs_list' : list of dict - List of prediction results, one
                   for each input sample
                - 'batch_size' : int - Number of predictions made
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"inputs_list": inputs_list},
            timeout=self.timeout * 2  # Longer timeout for batch
        )
        response.raise_for_status()
        return response.json()
    