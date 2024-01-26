import unittest2 as unittest
import tomography_toolkit_dev.generator_module as generators
import tomography_toolkit_dev.theory_module as theories
import tomography_toolkit_dev.discriminator_module as discriminators
from tomography_toolkit_dev.cfg.configurations_v0 import generator_config, training_config, theory_config, discriminator_config
import numpy as np
import torch

class generator_tests(unittest.TestCase):
    """
    generator_tests class to test the generator modules.
    It inherits from the class "unittest.TestCase" which offers
    all the assertion methods that are used to test the code.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(generator_tests, self).__init__(*args, **kwargs)
        ids = generators.list_registered_modules()
        theory = theories.make("torch_proxy_theory_v0", config=theory_config)
        discriminator = discriminators.make("torch_mlp_discriminator_v0", config=discriminator_config)
        self.modules = []
        for id in ids:
            self.modules.append(generators.make(id, config=generator_config, module_chain=[theory, discriminator]))
        #TODO: Replace with data module when available to have a generic torch/tf noise tensor
        self.noise = torch.normal(training_config['noise_mean'], training_config['noise_std'], size=(training_config['batch_size'],generator_config['input_size']))
        # Dummy Norms
        #TODO: Replace when data module is available
        self.real_norms = torch.tensor([1.]).repeat(training_config['batch_size']), torch.tensor([1.]).repeat(training_config['batch_size'])

    def test_forward(self):
        for module in self.modules:
            generator_output = module.forward(self.noise)
            generator_output = np.array(generator_output.detach())
            self.assertEqual(generator_output.shape, (training_config['batch_size'], generator_config['output_size']))

    def test_generate(self):
        for module in self.modules:
            generator_output = module.generate(self.noise)
            generator_output = np.array(generator_output.detach())
            self.assertEqual(generator_output.shape, (training_config['batch_size'], generator_config['output_size']))

    def test_train(self):
        for module in self.modules:
            loss = module.train(self.noise, self.real_norms)
            self.assertEqual(len(loss), 3)

if __name__ == "__main__":
    # You can run this file using:   python unittest_example.py
    unittest.main()