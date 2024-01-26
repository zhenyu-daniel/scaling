# SciDAC Quantom Collab Dev [version 0.1]
Dev repository for SciDAC Quantom project

# Code contribution proceedure
1. Create an issue or access assigned issue
2. Create a new branch using the issue id
3. When your code is complete and updated in the branch then make a pull request

More details are found here: [HOWTO_CONTRIBUTE.md](HOWTO_CONTRIBUTE.md)

# Python package to run the SciDAC workflow

## Software Requirement

- Python 3.7 
- Additional python packages are defined in the setup.py
- This document assumes you are running at the top directory

## Directory Organization

```
├── env.yaml                          : Conda setup file with package requirements
├── environment.tx                    : Extra pip install
├── setup.py                          : Python setup file with requirements files
├── README.md                         : Readme documentation
├── tomography_toolkit_dev
    ├── generator_module              : folder containing different versions/types of generators 
    ├── theory_module                 : folder containing theory modules
    ├── experiment_module             : folder containing experiment modules
    ├── discrimnator_module           : folder containing discriminator modules
    ├── workflow                      : folder containing workflow modules / drivers
    ├── core                          : folder containing base classes
    ├── cfg                           : folder containing configuration files
    ├── utils                         : folder containing supporting tools (e.g. monitoring)
    └── demos                         : folder containing demos for utils  
    

```

## Installing

- Clone code from repo
```
git clone https://github.com/quantom-collab/tomography_toolkit_dev.git
cd tomography_toolkit_dev
```

* Create default conda environment setup:
```
conda env create --file env.yaml (only once)
conda activate tomography_env (required every time you use the package)
```

- Install package in environment
```
pip install -e . (only once)
```

## Implementing (Updating) a new (existing) Module
The creation / alteration of SciDAC workflow module has to follow certain rules. In order to keep any module development consistent with these rules (and consistent with existing modules) you may do the following:
- From the top directory, enter the helper scripts in the utils folder:

```
cd tomography_toolkit_dev/utils/helper_scripts
```
- Run the follwoing command:

```
python torch_module_creator.py
```
- And follow the instructions. 

