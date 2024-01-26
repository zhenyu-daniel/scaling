import unittest2 as unittest
import tomography_toolkit_dev.discriminator_module as discriminators
from tomography_toolkit_dev.cfg.configurations_v0 import training_config, theory_config, discriminator_config
import numpy as np
import torch

class discriminator_tests(unittest.TestCase):
    """
    discriminator_tests class to test the discriminator modules.
    It inherits from the class "unittest.TestCase" which offers
    all the assertion methods that are used to test the code.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(discriminator_tests, self).__init__(*args, **kwargs)
        ids = discriminators.list_registered_modules()
        self.modules = []
        for id in ids:
            self.modules.append(discriminators.make(id, config=discriminator_config))
        self.dummy_input = torch.ones((training_config['batch_size'], theory_config['n_events']))

    def test_forward(self):
        for module in self.modules:
            discriminator_output = module.forward(self.dummy_input)
            discriminator_output = np.array(discriminator_output.detach())
            self.assertEqual(discriminator_output.shape, (training_config['batch_size'], 1))

    def test_train(self):
        for module in self.modules:
            loss = module.train(self.dummy_input, self.dummy_input)
            self.assertEqual(len(loss), 2)

if __name__ == "__main__":
    # You can run this file using:   python unittest_example.py
    unittest.main()