# Autoencoder (AE) Workflow for the Proxy App

This branch contains an Autoencoder implementation for proxy app workflow. Moreover, the autoencoder (and the GAN actually) may be ran via horovod. 

## Running the AE workflow on Mac M1 Machines
Horovod and M1 are not compatible, thus you can not utilize horovod + GPU on your Mac. However, you are able to run horovod + CPU on your Mac. 

* To use horovod together with CPU, simply run:
```
horovodrun -np N --gloo python ae_driver.py cpu
```
* where N denotes the number of workers you wish to utilize
* To run the AE workflow on the GPU, run:
```
python ae_driver.py
```

## Running the AE workflow with CUDA
For machines that provide cuda, you may utilize horovod (together with cuda) via:
```
horovodrun -np N python ae_driver.py
```
where N denotes the number of workers you wish to utilize

## Additional Information
The configuration file to run this workflow can be found in the cfg directory:
```
tomography_toolkit_dev/cfg/configurations_ae_hvd.py
```
The autoencoder model (which consists of the encoder stage only) is defined in the generator directory:
```
tomography_toolkit_dev/generator_module/Torch_MLP_Encoder_hvd.py
```

# GAN Workflow for the Proxy App

The GAN workflow with all its functionality can be run in this workflow with and without horovod. 

## Running without horovod

In order to run without horovod, simply do:
```
python driver.py
```

The correspinding configuration file is:
```
tomography_toolkit_dev/cfg/configurations_v0.py
```

## Running with horovod

To utilize horovod execute:
```
horovodrun -np N python driver_hvd.py
```
Where N denotes the number of workers. The underlying configuration file is:
```
tomography_toolkit_dev/cfg/configurations_hvd.py
```

## Overwriting Settings
Both driver scripts presented in this section allow to change certain configurstion settings by providing arguments to the driver script:
```
python driver.py arg1 arg2 arg3 arg4 arg5 (arg6)
```
with:
- arg1 : The name of the outputfile
- arg2 : The batch size
- arg3 : Number of events to select from the main data set (set this argument to -1 if you wish to utilize the full statistics)
- arg4 : Number of events to use in the post-training analysis 
- arg5 : Number of events to use in the snaphsot analysis 
- arg6 : This argument is optional and changes the location of the data set to be analyzed
