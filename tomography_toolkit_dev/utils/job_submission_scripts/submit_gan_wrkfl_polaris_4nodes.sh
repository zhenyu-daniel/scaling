#!/bin/bash -l                                                                                                            
#PBS -l select=4:ncpus=4:ngpus=4:system=polaris                                                                           
#PBS -l place=scatter                                                                                                     
#PBS -l walltime=05:30:00                                                                                                 
#PBS -l filesystems=home:grand                                                                                            
#PBS -q preemptable                                                                                                       
#PBS -A quantom_scidac                           

# This is an example script, trying to explain the usage of the quantom workflow on polaris, together with horovod
# This script will read the toy data from grand and write all results to: 'grand/quantom_scidac/results'
# Below are a few settings that you may or may not want to change.
# In order to check which settings can be changed via the command line, run:
# python driver_hvd.py -h (for the horovod workflow)

# To submit this script to the farm, run:
# qsub submit_gan_wrkfl_polaris_4nodes.sh

# Set the dirctory where the workflow: 'quantom_toolkit_dev' is stored:
workflowDir=/home/dlersch/SCIDAC

# Define the environment and where it can be found:
workflowEnv=/home/dlersch/envs/tomography_env

# Set the name of the folder where all results are stored:
# Important note: the workflow will automatically add the extension '_npN' to the folder name
# where N is the number of gpus
resultFolder="proxy_gan_on_polaris"

# Adjust the batch size:
batchSize=1000

# Set the total number of GPUs:
NGPUs=16

# Set the number of GPUs per node:
NGPUsPerServer=4

# Define a controller (e.g. gloo or mpi)
controller=gloo

# Output location (which is assumed to be grand and should not be changed when working on polaris)
outputLoc="/grand/quantom_scidac/results"

# Define the full path where the data is stored (which again is assumed to be on grand. please do not change this when working on polaris)
dataPath="/grand/quantom_scidac/sample_data/events_data.pkl.npy"

# This script assumes that four nodes with max. 4 GPUs are available, thus we have to make sure
# that: (a) NGPUs <= 16 and: (b) NGPUsPerServer * 4 == NGPUs

if [ $NGPUs -le 16 ] && [ $(($NGPUsPerServer*4)) -eq $NGPUs ]
then

    
    module load conda
    conda activate $workflowEnv

    echo Jobid: $PBS_JOBID
    echo Running on host `hostname`
    echo Running on nodes `cat $PBS_NODEFILE`
    server1=`sed -n '1p' $PBS_NODEFILE`
    server2=`sed -n '2p' $PBS_NODEFILE`
    server3=`sed -n '3p' $PBS_NODEFILE`
    server4=`sed -n '4p' $PBS_NODEFILE`

    serverSettings=$server1:$NGPUsPerServer,$server2:$NGPUsPerServer,$server3:$NGPUsPerServer,$server4:$NGPUsPerServer
    driverScript=$workflowDir/quantom_toolkit_dev/tomography_toolkit_dev/workflow/driver_hvd.py

    horovodrun -np $NGPUs -H $serverSettings --$controller python $driverScript --result_folder $resultFolder --batch_size $batchSize --output_loc $outputLoc --path_to_data $dataPath

else

   if [ $NGPUs -ge 16 ]
   then 
        echo "WARNING! Please make sure that: NGPUs <= 16"
   else 
        echo "WARNING! Please make sure that: NGPUsPerServer*4 == NGPUS"
    fi

fi
   
