import unittest2 as unittest
import tomography_toolkit_dev.theory_module as theories
from tomography_toolkit_dev.cfg.configurations_v0 import theory_config
import numpy as np
import torch

class theory_tests(unittest.TestCase):
    """
    theory_tests class to test the theory modules.
    It inherits from the class "unittest.TestCase" which offers
    all the assertion methods that are used to test the code.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(theory_tests, self).__init__(*args, **kwargs)
        ids = theories.list_registered_modules()
        self.modules = []
        for id in ids:
            self.modules.append(theories.make(id, config=theory_config))
        self.sample_params = torch.tensor(np.random.uniform(low=theory_config['parmin'], high=theory_config['parmax'], size=(100, 6)))

    def test_paramsToEventsMap(self):
        for module in self.modules:
            theory_output = module.paramsToEventsMap(self.sample_params, 10)
            self.assertEqual(len(theory_output), 3)
            events = np.array(theory_output[0].detach())
            self.assertEqual(events.shape, (100, 2, 10))

    def test_forward(self):
        for module in self.modules:
            theory_output = module.forward(self.sample_params)
            self.assertEqual(len(theory_output), 3)
            events = np.array(theory_output[0].detach())
            self.assertEqual(events.shape, (100, 2, 1))

if __name__ == "__main__":
    # You can run this file using:   python unittest_example.py
    unittest.main()