# Ensemble Offline Analysis Tool

This is a very simple script to analyze multiple GAN models that have been trained on the same data, using the same settings

## Running the Analysis

Before running the analysis, one needs to change the settings in:
```
run_ensemble_analyzer.py
```
The settings are explained and summarized in a dictionary called: cfg. After changing the settings accordingly, simply run:
```
python run_ensemble_analyzer.py
```
## Inspecting the Results

After running the command above, a folder with three subdirectories will be produced:
- plots: Three .png files that compare the individual and ensamble GAN performance
- raw_plots: The residuals and pdf distributions for each model
- npy_data: The ensemble predictions stored as .npy data

## Checking the Math

The critical calculations (and potential bugs) are done in:
```
ensemble_analyzer.py
```
