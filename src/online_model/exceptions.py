# exceptions.py
"""
Custom exceptions for the ML inference application.
"""

class OutputWriteFailure(Exception):
    """
    Exception raised when output variables cannot be written after retries.
    Signals that the iteration should be abandoned and restarted with fresh inputs.
    """
    pass