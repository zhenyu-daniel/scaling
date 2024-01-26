#!/usr/bin/env bash                                                                                                                                                        
#SBATCH --partition=gpu                                                                                                                                                    
#SBATCH --gres=gpu:T4:4                                                                                                                                                    
#SBATCH --time=24:00:00                                                                                                                                                    
#SBATCH --mem=64GB                                                                                                                                                         
#SBATCH --nodes=1                                                                                                                                                          
#SBATCH --job-name=TomographyHvdTest     

# This script tries to demonstrate the usage of the quantom workflow on the Jefferson Lab ifarm. 
# We can utilize a maximum of 4 GPUs. The script will run the sequential workflow, if one GPU is requested
# Otherwise, it will utilized horovod for parallelized training. Below are a few settings that might be interesting to change.
# In order to check which settings can be changed via the command line, run:
# python driver.py -h (for the sequential workflow)
# python driver_hvd.py -h (for the horovod workflow)

# To submit this script to the farm, run:
# sbatch submit_gan_wrkfl_jlab.sh


# Define the directory where the workflow ('quantom_toolkit_dev') is stored
workflowDir="/work/data_science/dlersch/SCIDAC"

# Define the environment that you are using:
workflowEnv=tomography_ifarm


# Define the number of GPUs you wish to utilize (this number can not be larger than 4)
NGPUs=4

if [ $NGPUs -le 4 ]
then
   
   source /etc/profile.d/modules.sh
   module use /apps/modulefiles
   module load anaconda3/4.5.12

   # Load the tomography env                                                                                                                                                  
   source activate $workflowEnv

   # Load cuda and mpi --> Needed for horovod                                                                                                                                 
   module load cuda/10.0
   module load mpi

   driverScript=$workflowDir/quantom_toolkit_dev/tomography_toolkit_dev/workflow/ae_driver.py
   horovodrun -np $NGPUs python $driverScript cuda

else
   echo "WARNING! You are requesting more than 4 gpus."
fi
