from ensemble_analyzer import EnsembleAnalyzer
import numpy as np

# This function translates the predicted parameters (ML based or not) to PDFs (Parton Density Functions)
# It requires two arguments:
# i) The parameters
# ii) The number of points we would like to plot / calculate
def get_ud(p,n_points):
    xmin, xmax = 0.1, 0.99999
    dx = (xmax - xmin) / n_points

    x_full_range = np.arange(xmin,xmax,dx)
    u = p[0]*np.power(x_full_range, p[1]) * np.power((1-x_full_range), p[2])
    d = p[3]*np.power(x_full_range, p[4]) * np.power((1-x_full_range), p[5])
    
    return u, d, x_full_range


# Configuration needed to run the ensemble analyzer:
cfg = {
    'true_params': [0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8], #--> True parameters that we are trying to predict
    'n_epochs': 50000, #--> Number of (fit) iterations
    'observable_names': ['u','d'], #--> Names of the pdfs that we are trying to extract / fit
    'param_to_pdf_fn': get_ud, #--> Function that translates parameters to pdf
    'ensemble_names': ['jlab_results/gan_ana'+str(k) + '_1k_sample1_np4' for k in range(1,11)], #--> List of all models that have been trained to predict the parameters
    'param_plot_ylim': [-0.35,0.35], #--> Cosmetics: Range of the y-axis to plot the residuals
    'pdf_plot_ylim': [], #--> Cosmetics: Range of the y-axis to plot the pdfs
    'plot_header_extension': ' for N(events) = 1k', #--> Additonal name to put on the header
    'label_name': 'GAN Ensemble', #--> Add labels to the figures
    'result_folder': 'results_sample1_1k' #--> Name the result folder
}

analyzer = EnsembleAnalyzer(cfg)
analyzer.run()