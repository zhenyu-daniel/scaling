import unittest2 as unittest

import tomography_toolkit_dev.generator_module as generators
import tomography_toolkit_dev.discriminator_module as discriminators
import tomography_toolkit_dev.theory_module as theories
import tomography_toolkit_dev.experimental_module as experiments
import tomography_toolkit_dev.workflow as workflows

from tomography_toolkit_dev.cfg.configurations_v0 import training_config, generator_config, discriminator_config, module_names, theory_config, experimental_config, data_config


class RegistryTests(unittest.TestCase):
    """
    Registry Test class to test all the registered modules are loaded properly.
    """
    def test_generators(self):
        """ 
        """
        generator_instance = generators.make("torch_mlp_generator_v0", config=generator_config, module_chain=[])
        self.assertEqual(generator_instance.name, "Torch MLP Generator Model")

    def test_discriminators(self):
        """ 
        """
        discriminator_instance = discriminators.make("torch_mlp_discriminator_v0", config=discriminator_config)
        self.assertEqual(discriminator_instance.name, "Torch MLP Discriminator Model")
    
    def test_theories(self):
        """ 
        """
        thoery_instance = theories.make("torch_proxy_theory_v0", config=theory_config)
        self.assertEqual(thoery_instance.name, "Torch Proxy Theory")

    def test_experiments(self):
        """
        """
        pass

    def test_workflows(self):
        """
        """
        workflow_instance = workflows.make("sequencial_workflow_v0", module_names=module_names, 
                                                            generator_config=generator_config, 
                                                            discriminator_config=discriminator_config, 
                                                            theory_config=theory_config, 
                                                            experimental_config=experimental_config, 
                                                            training_config=training_config, 
                                                            data_config=data_config)
        self.assertEqual(workflow_instance.name, "sequential_workflow_v0")
        


if __name__ == '__main__':
    unittest.main()

