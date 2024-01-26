#!/usr/bin/env bash

ensembleStart=${1}
ensembleEnd=${2}

# BASIC SETTINGS:
workflowDir="/work/data_science/dlersch/SciDAC/Ensemble_Analysis"
env=tomography_ifarm
batchDim=1000
outputDir=$workflowDir

for ((i=$ensembleStart; i<=$ensembleEnd; i++))
do
  resultFolder="gan_ensemble_ana"${i}
  logFileName=./gan_ens${i}.log

  sbatch --output=$logFileName run_gan_wrkfl_at_jlab.sh $workflowDir $env $batchDim $resultFolder $outputDir

done