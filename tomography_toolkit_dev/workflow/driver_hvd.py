import torch
from tomography_toolkit_dev.cfg.configurations_hvd import training_config, generator_config, discriminator_config, module_names, theory_config, experimental_config, data_config
import tomography_toolkit_dev.workflow as workflows


if __name__ == '__main__':

   devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
   print("Devices available for training: ", devices)

   wflow = workflows.make("horovod_workflow_v0", module_names=module_names, 
                                                            generator_config=generator_config, 
                                                            discriminator_config=discriminator_config, 
                                                            theory_config=theory_config, 
                                                            experimental_config=experimental_config, 
                                                            training_config=training_config, 
                                                            data_config=data_config,
                                                              devices=devices)

   wflow.run()