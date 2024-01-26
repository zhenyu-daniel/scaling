#!/usr/bin/env bash                                                                                                                                                        
#SBATCH --partition=gpu                                                                                                                                                    
#SBATCH --gres=gpu:T4:4                                                                                                                                                    
#SBATCH --time=24:00:00                                                                                                                                                    
#SBATCH --mem=64GB                                                                                                                                                         
#SBATCH --nodes=1                                                                                                                                                          
#SBATCH --job-name=SynchTest     

# This script tries to demonstrate the usage of the quantom workflow on the Jefferson Lab ifarm. 
# We can utilize a maximum of 4 GPUs. The script will run the sequential workflow, if one GPU is requested
# Otherwise, it will utilized horovod for parallelized training. Below are a few settings that might be interesting to change.
# In order to check which settings can be changed via the command line, run:
# python driver.py -h (for the sequential workflow)
# python driver_hvd.py -h (for the horovod workflow)

# To submit this script to the farm, run:
# sbatch submit_gan_wrkfl_jlab.sh


# Define the directory where the workflow ('quantom_toolkit_dev') is stored
workflowDir=${1}

# Define the environment that you are using:
workflowEnv=${2}

# Set the batch size you wish to use:
batchSize=${3}

# Define the name of the folder where all results will be stored:
# The workflow will automatically add the extension '_npN' if N>1 gpus are used
resultFolder=${4}

# Define the output location where the result folder will be written to:
outputLoc=${5}

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

   if [ $NGPUs -gt 1 ]
   then 
     # Run with multiple GPUs: (--> Use horovod)
     driverScript=$workflowDir/quantom_toolkit_dev/tomography_toolkit_dev/workflow/driver_hvd.py
     horovodrun -np $NGPUs python $driverScript --num_epochs 20000 --snapshot_epoch 10 --batch_size $batchSize --result_folder $resultFolder --output_loc $outputLoc --use_timestamp --store_models_only

   else
     # Run with just one GPU: (--> use sequential)
     driverScript=$workflowDir/quantom_toolkit_dev/tomography_toolkit_dev/workflow/driver.py
     python $driverScript --num_epochs 20000 --snapshot_epoch 10 --batch_size $batchSize --result_folder $resultFolder --output_loc $outputLoc --use_timestamp --store_models_only

   fi

else
   echo "WARNING! You are requesting more than 4 gpus."
fi
