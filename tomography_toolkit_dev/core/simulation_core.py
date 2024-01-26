from abc import ABC, abstractmethod


class Simulation(ABC):
    """
    Generator base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def apply_detector_response(self,data):
        return NotImplemented
        
    # Apply function during training:
    @abstractmethod
    def forward(self,data):
        return NotImplemented
    
    # Apply function outside of training loop:
    @abstractmethod
    def apply(self,data):
        return NotImplemented

