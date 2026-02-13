import requests
from typing import Dict, Any, List, Optional
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for calling the inference service
    Uses connection pooling for efficient HTTP requests.
    
    Parameters
    ----------
    base_url : str
        Base URL of the inference service (e.g., "http://inference-service:8000").
    timeout : int, optional
        Request timeout in seconds. Default is 30.
    max_retries : int, optional
        Maximum number of retries for failed requests. Default is 3.
    pool_connections : int, optional
        Number of connection pools to cache. Default is 10.
    pool_maxsize : int, optional
        Maximum number of connections to save in the pool. Default is 10.
    """
    
    def __init__(
        self, 
        base_url: str, 
        timeout: int = 30,
        max_retries: int = 3,
        pool_connections: int = 10,
        pool_maxsize: int = 10
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,  # Wait 0.5s, 1s, 2s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )
        
        # Mount adapter for both http and https
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers (avoid connection close)
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Initialized client for {self.base_url} with connection pooling")

    def __del__(self):
        """Cleanup: Close session on deletion"""
        if hasattr(self, 'session'):
            self.session.close()

    
    def health_check(self) -> bool:
        """
        Check if the inference service is healthy.
        
        Returns
        -------
        bool
            True if the service responds with status 200, False otherwise.
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
        """Get model metadata"""
        response = self.session.get(
            f"{self.base_url}/model/info",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_inputs(self) -> Dict[str, Any]:
        """Get model input specifications"""
        response = self.session.get(
            f"{self.base_url}/inputs",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_outputs(self) -> Dict[str, Any]:
        """Get model output specifications"""
        response = self.session.get(
            f"{self.base_url}/outputs",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a prediction
        
        Args:
            inputs: Dictionary of input features
            
        Returns:
            Prediction response with outputs
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json={"inputs": inputs},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, inputs_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Make batch predictions
        
        Args:
            inputs_list: List of input dictionaries (each can be partial)
            
        Returns:
            Batch prediction response with list of outputs
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"inputs_list": inputs_list},
            timeout=self.timeout * 2  # Longer timeout for batch
        )
        response.raise_for_status()
        return response.json()