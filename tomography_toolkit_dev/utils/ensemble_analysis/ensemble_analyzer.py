import torch
import numpy as np
import os
import matplotlib.pyplot as plt

class EnsembleAnalyzer(object):

    # Initialize:
    #******************************
    def __init__(self,config):
        
        self.ensemble_names = config['ensemble_names'] if 'ensemble_names' in config else None
        self.true_params_list = config['true_params'] if 'true_params' in config else None
        self.observable_names = config['observable_names'] if 'observable_names' in config else None
        self.param_to_pdf_fn = config['param_to_pdf_fn'] if 'param_to_pdf_fn' in config else None
        self.result_folder_name = config['result_folder'] if 'result_folder' in config else 'results'

        # Plotting:
        self.n_pdf_points = config['n_pdf_points'] if 'n_pdf_points' in config else 100
        self.font_size = config['font_size'] if 'font_size' in config else 20
        self.leg_font_size = config['leg_font_size'] if 'leg_font_size' in config else 15
        self.max_hist_warning = config['max_hist_warning'] if 'max_hist_warning' in config else 100
        self.n_bins = config['n_bins'] if 'n_bins' in config else 100
        self.n_epochs = config['n_epochs'] if 'n_epochs' in config else 100
        self.snapshot_epoch = config['snapshot_epoch'] if 'snapshot_epoch' in config else 10
        self.snapshot_dir_name = config['snapshot_dir_name'] if 'snapshot_dir_name' in config else 'snapshot_npy'
        self.plot_header_extension = config['plot_header_extension'] if 'plot_header_extension' in config else ''
        self.param_plot_ylim = config['param_plot_ylim'] if 'param_plot_ylim' in config else [-0.25,0.25]
        self.pdf_plot_ylim = config['pdf_plot_ylim'] if 'pdf_plot_ylim' in config else [-0.25,0.25]
        self.label_name = config['label_name'] if 'label_name' in config else 'Ensemble'

        self.ensemble_size = len(self.ensemble_names)
        self.n_observables = len(self.observable_names)

        plt.rcParams.update({'font.size':self.font_size})
        plt.rcParams.update({'figure.max_open_warning':self.max_hist_warning})
    #******************************

    # Get true information first:
    #******************************
    def gather_true_info(self):
        self.true_params = np.array(self.true_params_list)
        self.true_pdf_data = self.param_to_pdf_fn(self.true_params,self.n_pdf_points)
    #******************************

    # Create result folder which contains plots and data:
    #******************************
    def create_result_folders(self):
        # Main folder:
        if os.path.exists(self.result_folder_name) == False:
            os.mkdir(self.result_folder_name)

        # Folder for nice plots:
        self.plot_dir = self.result_folder_name + '/plots'
        if os.path.exists(self.plot_dir) == False:
            os.mkdir(self.plot_dir)

        # Folder for the raw parameter residual / pdf plots:
        self.raw_plot_dir = self.result_folder_name + '/raw_plots'
        if os.path.exists(self.raw_plot_dir) == False:
            os.mkdir(self.raw_plot_dir)

        # Folder for storing the ensemble results as .npy data:
        self.npy_dir = self.result_folder_name + '/npy_ensemble_data'
        if os.path.exists(self.npy_dir) == False:
            os.mkdir(self.npy_dir)
    #******************************

    # Scan the snaphsot folder for any npy data:
    #******************************
    def scan_snapshot_folder(self):
        # We just scan the folder for one model only:
        folder_name = self.ensemble_names[0] + '/' + self.snapshot_dir_name

        it = []
        self.extensions = []
        #++++++++++++++++++++++++++++
        for epoch in range(1,1+self.n_epochs):
            epoch_str = str(epoch) + 'epochs'
            epoch_str = epoch_str.zfill(6 + len(str(self.n_epochs))) 

            current_extension = 'parameter_residuals_' + epoch_str
            param_name = folder_name + '/' + current_extension

            if os.path.exists(param_name+".npy"):
                it.append(epoch)
                self.extensions.append(current_extension)
        #++++++++++++++++++++++++++++

        self.epochs = np.array(it)
    #******************************

    # Set up monitoring plots:
    #****************************** 
    def setup_monitoring_plots(self):
        self.n_snapshots = self.epochs.shape[0]
        self.n_parameters = self.true_params.shape[0]

        # Define parameter residual monitoring plots:
        self.param_residual_plots = plt.subplots(self.n_parameters,1,figsize=(12,8),sharex=True)
        self.param_residual_plots[0].subplots_adjust(wspace=0.35)
        self.param_residual_plots[0].suptitle('Parameter Residuals (R)' + self.plot_header_extension)

        # Raw plots, i.e. plotting the residuals / PDFs directly:
        self.raw_residual_plots = []
        self.raw_pdf_plots = []
        #++++++++++++++++++++++++++++++++
        for _ in range(self.n_snapshots):
            current_plots = plt.subplots(1,self.n_parameters,figsize=(17,8),sharey=True)
            self.raw_residual_plots.append(current_plots)

            current_plots = plt.subplots(1,self.n_observables,figsize=(12,8))
            current_plots[0].subplots_adjust(hspace=0.5)
            self.raw_pdf_plots.append(current_plots)
        #++++++++++++++++++++++++++++++++

        # Define pdf residual monitoring plots:
        self.pdf_residual_plots = plt.subplots(self.n_observables,1,figsize=(12,8),sharex=True)
        self.pdf_residual_plots[0].suptitle('PDF Residuals' + self.plot_header_extension)

        # Highlight the errors of the ensemble predictions:
        self.pdf_err_plots = plt.subplots(self.n_observables,1,figsize=(12,8),sharex=True)
        self.pdf_err_plots[0].suptitle('PDF Residuals (Ensemble only)' + self.plot_header_extension)
    #******************************  

    # Reconstruct the analysis from npy data:
    #****************************** 
    def reconstruct_analysis_from_npy(self):
        # Set up a dictionary to collect all predicted parameters and to form an ensemble
        self.ensemble_parameter_dict = {}
   
        # Loop over models in ensemble:
        #+++++++++++++++++++++++++++++++++++++++++++
        for m in range(self.ensemble_size):
            folder_name = self.ensemble_names[m] + '/' + self.snapshot_dir_name

            # i) Collect parameter residual mean and std. dev. for each model:
            param_residual_mean = np.zeros((self.n_snapshots,self.n_parameters))
            param_residual_std = np.zeros((self.n_snapshots,self.n_parameters))

            # ii) Collect pdf (parton density function) residual mean and std. dev.: (--> Not to be mistaken with probability density function)
            pdf_residual_mean = np.zeros((self.n_snapshots,self.n_observables))
            pdf_residual_std = np.zeros((self.n_snapshots,self.n_observables))
            
            # iii) Initialize ensemble collection:
            self.ensemble_parameter_dict[m] = []
            
            # Loop over all detected snapshots:
            #+++++++++++++++++++++++++++++++++++++++++++
            for s in range(self.n_snapshots): 
                parameter_data = folder_name + '/' + self.extensions[s] + '.npy'

                if os.path.exists(parameter_data):
                   
                   # i) Parameters: 
                   parameter_residuals = np.load(parameter_data)
                   residuals_mean = np.mean(parameter_residuals,0)
                   residuals_std = np.std(parameter_residuals,0)
                   
                   #+++++++++++++++++++++++++++++++++++++++++++
                   for p in range(self.n_parameters):
                       param_residual_mean[s][p] = residuals_mean[p]
                       param_residual_std[s][p] = residuals_std[p]

                       # Fill the raw plots:
                       self.raw_residual_plots[s][1][p].hist(parameter_residuals[:,p],self.n_bins,histtype='step',linewidth=3.0)
                   #+++++++++++++++++++++++++++++++++++++++++++

                   # ii) PDFs (parton density functions)
                   predicted_params = self.true_params + parameter_residuals #--> Extract the predicted parameters from the stored residuals:
                   pred_pdf_data = np.apply_along_axis(self.param_to_pdf_fn,1,predicted_params,self.n_pdf_points) #--> It is assumed that the function returns the PDFs + x-values
                   # So if we have two PDFs (Parton Density Functions) and one x-value, we would expect this function to return three arguments
                   # Each predicted parameter set produces three arguments and each argument has the dimension: n_points.
                   # Thus, the output of this function is: Dim(Parameters) x (#PDFs + #x-values) x n_points
                   pdf_residuals = self.true_pdf_data - pred_pdf_data
                   pdf_mu = np.mean(pdf_residuals,axis=0)
                   pdf_sigma = np.std(pdf_residuals,axis=0)

                   #+++++++++++++++++++
                   for u in range(self.n_observables):
                       pdf_residual_mean[s][u] = np.mean(pdf_mu[u])
                       pdf_residual_std[s][u] = np.mean(pdf_sigma[u])

                       self.raw_pdf_plots[s][1][u].plot(np.mean(pred_pdf_data,0)[2],np.mean(pred_pdf_data,0)[u],linewidth=3.0)
                   #+++++++++++++++++++

                   # iii) Collect parameter (residuals) for ensemble:
                   self.ensemble_parameter_dict[m].append(parameter_residuals)
                   
                else:
                    print(" ")
                    print(">>> WARNING: npy data for model: " + str(m) + " and epoch " + str(self.epochs[s]) + " does not exst! <<<")
                    print(">>> Please make sure that the collected npy data is consistent throughout all models within the ensembke <<<")
                    print("  ")

                    break
            #+++++++++++++++++++++++++++++++++++++++++++

            # Fill monitoring plots:
            # i) Residuals:
            #+++++++++++++++++++++++++
            for p in range(self.n_parameters):
                self.param_residual_plots[1][p].errorbar(x=self.epochs,y=param_residual_mean[:,p],yerr=param_residual_std[:,p],linewidth=3.0)
                j=p+1
                self.param_residual_plots[1][p].set_ylabel('R' + str(j))
                self.param_residual_plots[1][p].grid(True)

                if self.param_plot_ylim is not None and len(self.param_plot_ylim) == 2:
                  self.param_residual_plots[1][p].set_ylim(self.param_plot_ylim[0],self.param_plot_ylim[1])
            #+++++++++++++++++++++++++
            self.param_residual_plots[1][self.n_parameters-1].set_xlabel('Epoch')

            # ii) Observables:
            #+++++++++++++++++++++++++
            for u in range(self.n_observables):
                self.pdf_residual_plots[1][u].errorbar(x=self.epochs,y=pdf_residual_mean[:,u],yerr=pdf_residual_std[:,u],linewidth=3.0)
                self.pdf_residual_plots[1][u].set_ylabel('Residual on ' + self.observable_names[u])
                self.pdf_residual_plots[1][u].grid(True)

                if self.pdf_plot_ylim is not None and len(self.pdf_plot_ylim) == 2:
                   self.pdf_residual_plots[1][u].set_ylim(self.pdf_plot_ylim[0],self.pdf_plot_ylim[1])
            #+++++++++++++++++++++++++
            self.pdf_residual_plots[1][self.n_observables-1].set_xlabel('Epoch')
        #+++++++++++++++++++++++++++++++++++++++++++

    #****************************** 

    # Extract inforamtion from ensemble:
    #****************************** 
    def extract_information_from_ensemble(self):
        # i) Ensemble residuals:
        ensemble_residual_mean = np.zeros((self.n_snapshots,self.n_parameters))
        ensemble_residual_std = np.zeros((self.n_snapshots,self.n_parameters))

        # ii) Ensemble PDFs (parton density functions)
        ensemble_pdf_residual_mean = np.zeros((self.n_snapshots,self.n_observables))
        ensemble_pdf_residual_std = np.zeros((self.n_snapshots,self.n_observables))

        #+++++++++++++++++++++++++++++++
        for s in range(self.n_snapshots):

            # i) Parameter residuals:

            residual_list = []
            # Retreive information from the dictionary we filled ealrier:
            #+++++++++++++++++++++++++++++++
            for m in range(self.ensemble_size):
                current_residual = self.ensemble_parameter_dict[m][s]
                residual_list.append(current_residual)
            #+++++++++++++++++++++++++++++++
            ensemble_residuals = np.concatenate(residual_list,0)
            
            # Get the residuals information:
            #+++++++++++++++++++++++++++++++
            for p in range(self.n_parameters):
                ensemble_residual_mean[s][p] = np.mean(ensemble_residuals,0)[p]
                ensemble_residual_std[s][p] = np.std(ensemble_residuals,0)[p]

                # Set raw residual plots:
                #self.raw_residual_plots[s][1][p].hist(ensemble_residuals[:,p],self.n_bins,histtype='step',color='k',linewidth=3.0,label=self.label_name)
                j = p+1
                self.raw_residual_plots[s][1][p].set_xlabel('R'+str(j))
                self.raw_residual_plots[s][1][p].grid(True)
                #self.raw_residual_plots[s][1][p].legend(fontsize=self.leg_font_size)
            #+++++++++++++++++++++++++++++++
            self.raw_residual_plots[s][1][0].set_ylabel('Entries')


            # ii) PDF (parton density function) residuals:
            ensemble_predicted_params = self.true_params + ensemble_residuals
            ensemble_pdf_data = np.apply_along_axis(self.param_to_pdf_fn,1,ensemble_predicted_params,self.n_pdf_points)
            ensemble_pdf_residuals = self.true_pdf_data - ensemble_pdf_data
    
            ens_pdf_mu = np.mean(ensemble_pdf_residuals,axis=0)
            ens_pdf_sigma = np.std(ensemble_pdf_residuals,axis=0)

            #+++++++++++++++++++
            for u in range(self.n_observables):
                ensemble_pdf_residual_mean[s][u] = np.mean(ens_pdf_mu[u])
                ensemble_pdf_residual_std[s][u] = np.mean(ens_pdf_sigma[u])

                #self.raw_pdf_plots[s][1][u].plot(np.mean(ensemble_pdf_data,0)[2],np.mean(ensemble_pdf_data,0)[u],'k',linewidth=3.0,label=self.label_name)
                self.raw_pdf_plots[s][1][u].set_xlabel('x')
                self.raw_pdf_plots[s][1][u].set_ylabel('PDF(' + self.observable_names[u]+')')
                self.raw_pdf_plots[s][1][u].grid(True)
                #self.raw_pdf_plots[s][1][u].legend(fontsize=self.leg_font_size)
            #+++++++++++++++++++

            # Write raw plots to file:
            self.raw_residual_plots[s][0].savefig(self.raw_plot_dir+'/raw_param_residuals_' + self.extensions[s]+'.png')
            plt.close(self.raw_residual_plots[s][0])
            self.raw_pdf_plots[s][0].savefig(self.raw_plot_dir+'/raw_pdf_residuals_' + self.extensions[s]+'.png')
            plt.close(self.raw_pdf_plots[s][0])
        #+++++++++++++++++++++++++++++++

        # Now we can add the ensemble results to the plots we filled earlier:
        #+++++++++++++++++++++++++++++++
        for p in range(self.n_parameters):
            self.param_residual_plots[1][p].errorbar(x=self.epochs,y=ensemble_residual_mean[:,p],yerr=ensemble_residual_std[:,p],fmt='ko',linewidth=3.0,capsize=5.0,label=self.label_name)
            self.param_residual_plots[1][p].legend(fontsize=self.leg_font_size)
        #+++++++++++++++++++++++++++++++

        #+++++++++++++++++++++++++++++++
        for u in range(self.n_observables):
            self.pdf_residual_plots[1][u].errorbar(x=self.epochs,y=ensemble_pdf_residual_mean[:,u],yerr=ensemble_pdf_residual_std[:,u],fmt='ko',linewidth=3.0,capsize=5.0,label=self.label_name)
            self.pdf_residual_plots[1][u].legend(fontsize=self.leg_font_size)

            self.pdf_err_plots[1][u].errorbar(x=self.epochs,y=ensemble_pdf_residual_mean[:,u],yerr=ensemble_pdf_residual_std[:,u],fmt='ko',linewidth=3.0,capsize=5.0,label=self.label_name)
            self.pdf_err_plots[1][u].set_ylabel('Residual on ' + str(self.observable_names[u]))
            self.pdf_err_plots[1][u].grid(True)
            self.pdf_err_plots[1][u].legend(fontsize=self.leg_font_size)

            ax_err = self.pdf_err_plots[1][u].twinx()
            ax_err.set_ylabel(r'$\sigma($' + self.observable_names[u] + ') Ensemble',color='tab:red')
            ax_err.plot(self.epochs,ensemble_pdf_residual_std[:,u],color='tab:red',linewidth=3.0)
            ax_err.tick_params(axis='y',labelcolor='tab:red')
        #+++++++++++++++++++++++++++++++
        self.pdf_err_plots[1][self.n_observables-1].set_xlabel('Epoch')

        self.param_residual_plots[0].savefig(self.plot_dir+'/param_residuals.png')
        plt.close(self.param_residual_plots[0])
        self.pdf_residual_plots[0].savefig(self.plot_dir+'/pdf_residuals.png')
        plt.close(self.pdf_residual_plots[0])
        self.pdf_err_plots[0].savefig(self.plot_dir+'/pdf_residuals_err.png')
        plt.close(self.pdf_err_plots[0])

        # Store ensemble results in npy format:
        np.save(self.npy_dir+'/ensemble_parameters.npy',ensemble_residuals)
        np.save(self.npy_dir+'/ensemble_pdfs.npy',np.mean(ensemble_pdf_data,0))
    #******************************


    # Put everything togehter:
    #****************************** 
    def run(self):
        # Get true info:
        print("Retreive true parameters / distributions...")

        self.gather_true_info()

        print("...done!")
        print(" ") 

        # Scan snapshot folder:
        print("Scan snapshot folder for existing data...")
        
        self.scan_snapshot_folder()

        print("...done!")
        print(" ")

        if len(self.extensions) > 0:

            # Set up folder to store plots and such:
            print("Create result folder(s)...")

            self.create_result_folders()

            print("...done!")
            print(" ")

            # Set up monitoring plots:
            print("Define monitoring plots...")

            self.setup_monitoring_plots()

            print("...done!")
            print(" ")


            # Reconstruct analysis:
            print("Run analysis reconstruction from npy data...")
            
            self.reconstruct_analysis_from_npy()

            print("...done!")
            print(" ")

            # Get info from ensemble:
            print("Extract results from ensemble...")
   
            self.extract_information_from_ensemble()

            print("...done!")
            print(" ")

        else:
            print(">>> Error! No data detected! Going to stop analysis here <<<")
            print(" ")
    #****************************** 
  

    