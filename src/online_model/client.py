import requests
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for calling the inference service"""
    
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
        """Check if service is healthy"""
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
    
    def load_model(self, model_name: str, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a different model
        
        Args:
            model_name: Name of the model to load
            model_version: Version or stage (optional)
        
        Returns:
            Load response with status
        """
        payload = {"model_name": model_name}
        if model_version:
            payload["model_version"] = model_version
        
        response = self.session.post(
            f"{self.base_url}/model/load",
            json=payload,
            timeout=60  # Model loading can take longer
        )
        response.raise_for_status()
        return response.json()