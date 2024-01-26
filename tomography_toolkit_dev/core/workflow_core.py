from abc import ABC, abstractmethod


class Workflow(ABC):
    """
    Generator base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self):
        return NotImplemented