from abc import ABC, abstractmethod


class Theory(ABC):
    """
    Theory base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).
    """

    def __init__(self, **kwargs):
        """
            Initialization of various parameters required for events generation.
        Input
        -----
            A config file containing various parameter values to initialize
        """
        pass

    @abstractmethod
    def paramsToEventsMap(self):
        """
            Function to generate events from parameters
        Inputs
        ------
            Params: numpy nd-array or torch tensor
                array/tensor containing n parameters to generate events for
            nevents: int
                number of events to generate per parameter set
        
        Outputs
        -------
            events: tensor
                A tensor containing generated events
            
            norms: tensor
                Normalizing constants
        """
        raise NotImplemented

    @abstractmethod
    def forward(self, parameters):
        """
            Function to generate events from parameters during a forward pass. It just calls paramsToEventsMap function above!
        """
        raise NotImplemented
        
    @abstractmethod
    def apply(self, parameters):
        """
            Exact same function as .forward(). But this one is supposed to run 'outside' the training loop.
            While this feature might be needed for some cases, there might be applications that act 'different' under
            training conditions, mainly because of differentiability
        """
        raise NotImplemented
