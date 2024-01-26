import torch
from tomography_toolkit_dev.cfg.configurations_ae_hvd import training_config, generator_config, module_names, theory_config, experimental_config, data_config
import tomography_toolkit_dev.workflow as workflows
import sys


if __name__ == '__main__':
   
   device_arg = "cpu"
   if torch.cuda.is_available():
      device_arg = "cuda"
   elif torch.backends.mps.is_available():
      device_arg = "mps"

   if len(sys.argv) > 1:
      device_arg = sys.argv[1]

   devices = torch.device(device_arg) 
   print("Devices available for training: ", devices)
   
   if device_arg != "mps":
      wflow = workflows.make("horovod_workflow_v1", module_names=module_names, 
                                                            generator_config=generator_config, 
                                                            theory_config=theory_config, 
                                                            experimental_config=experimental_config, 
                                                            training_config=training_config, 
                                                            data_config=data_config,
                                                            devices=devices)
      
   else:
      wflow = workflows.make("sequential_workflow_v1", module_names=module_names, 
                                                            generator_config=generator_config, 
                                                            theory_config=theory_config, 
                                                            experimental_config=experimental_config, 
                                                            training_config=training_config, 
                                                            data_config=data_config,
                                                            devices=devices)

   wflow.run()