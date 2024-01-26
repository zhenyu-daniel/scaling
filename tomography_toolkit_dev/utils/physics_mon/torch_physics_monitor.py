import torch
import functorch
import numpy as np
import matplotlib.pyplot as plt

class Physics_Monitor(object):

    # INITIALIZE

    #**********************
    def __init__(self,params_to_physics_func,generator,true_params,noise_mean,noise_std,config={},device='cpu'):
        self.params_to_physics_func = params_to_physics_func 
        self.generator = generator
        self.true_params = true_params
        self.n_parameters = true_params.size()[0]
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        
        # Settings for the plots:
        self.config = config
        self.font_size = self.config['font_size'] if 'font_size' in self.config else 20 
        self.line_width = self.config['line_width'] if 'line_width' in self.config else 2.0
        self.fig_width_scale = self.config['fig_width_scale'] if 'fig_width_scale' in self.config else 1.0
        self.fig_height_scale = self.config['fig_height_scale'] if 'fig_height_scale' in self.config else 1.0
        self.fig_wspace = self.config['fig_wspace'] if 'fig_wspace' in self.config else 0.5
        self.fig_hspace = self.config['fig_hspace'] if 'fig_hspace' in self.config else 0.25
        self.legend_font_size = self.config['legend_font_size'] if 'legend_font_size' in self.config else 15
        self.dist_names = self.config['dist_names'] if 'dist_names' in self.config else None
        self.min_q = self.config['min_q'] if 'min_q' in self.config else 0.0
        self.max_q = self.config['max_q'] if 'max_q' in self.config else 1.0
        self.x_labels = self.config['x_labels'] if 'x_labels' in self.config else []
        self.y_labels = self.config['y_labels'] if 'y_labels' in self.config else []
        self.pdf_format = self.config['pdf_format'] if 'pdf_format' in self.config else [1,1]


        plt.rcParams.update({'font.size':self.font_size})
        self.device = device
    #**********************

    # CREATE TRUE DISTRIBUTIONS

    #**********************
    def create_true_distributions(self):
        true_data = self.params_to_physics_func(self.true_params)

        true_npy_data = ()
        #+++++++++++++++++++++
        for el in true_data:
            true_npy_data = true_npy_data + (el.cpu().numpy(),)
        #+++++++++++++++++++++

        return true_npy_data
    #**********************

    # CREATE THE GENERATED DISTRIBUTIONS:

    #**********************
    def create_gen_distributions(self,n_samples,x=None):
        gen_params = None
        with torch.no_grad():
           if x is not None:
               gen_params = self.generator.generate(x).to(self.device)
           else:
               noise = torch.normal(mean=self.noise_mean,std=self.noise_std,size=(n_samples,self.n_parameters)).to(self.device)
               gen_params = self.generator.generate(noise).to(self.device)

           generated_data = functorch.vmap(self.params_to_physics_func)(gen_params)
        
           generated_npy_data = ()
           #+++++++++++++++++++++
           for el in generated_data:
              generated_npy_data = generated_npy_data + (el.cpu().numpy(),)
           #+++++++++++++++++++++

           return generated_npy_data
    #**********************

    # VISUALIZE THE TRUE AND GENERATED DISTRIBUTIONS:

    #**********************
    def compare_true_and_generated_distributions(self,true_npy_data,generated_npy_data):
        x = true_npy_data[-1]
        n_dists = len(true_npy_data) - 1

        fig_width = int(self.fig_width_scale * self.pdf_format[1] * 2 + 10)
        fig_height = int(self.fig_height_scale * self.pdf_format[0] * 2 + 6)

        plot_dict = {}

        # TODO: The following implementation is not pretty and needs to be optimized in the near future.

        # Store pdf comparison as single plots:
        if self.pdf_format[0] == 1 and self.pdf_format[1] == 1:
           #++++++++++++++++++++++++++
           for i in range(n_dists):
               f = i + 1

               if self.dist_names is not None:
                  f = self.dist_names[i]

               fig, ax = plt.subplots(figsize=(fig_width,fig_height))

               ax.plot(x,true_npy_data[i],'b-',linewidth=self.line_width,label='True')

               # The following has been taken from Kishans plotting tool:
               core_value = np.mean(generated_npy_data[i],axis=0)
               lower_limit = np.quantile(generated_npy_data[i],self.min_q,axis=0)
               upper_limit = np.quantile(generated_npy_data[i],self.max_q,axis=0)

               ax.plot(x,core_value,'r--',linewidth=self.line_width,label='GAN Prediction')
               ax.fill_between(x,lower_limit,upper_limit, alpha=0.25, color="red")
               ax.grid(True)
               ax.legend(fontsize=self.legend_font_size)
               ax.set_xlabel(self.x_labels[i])
               ax.set_ylabel(self.y_labels[i])

               key_name = 'true_vs_generated_pdf_' + str(f)
               plot_dict[key_name] = [fig,ax]
               plt.close()
           #++++++++++++++++++++++++++
        else:

            # Store pdfs under a different format: 
            fig, ax = plt.subplots(self.pdf_format[0],self.pdf_format[1],figsize=(fig_width,fig_height))
            fig.subplots_adjust(wspace=self.fig_wspace,hspace=self.fig_hspace)
            pdf_index = 0

            current_axis = None
            #++++++++++++++++++++++++
            for row in range(self.pdf_format[0]):
                #++++++++++++++++++++++++
                for col in range(self.pdf_format[1]):
                    f = pdf_index +1
                    if self.dist_names is not None:
                          f = self.dist_names[pdf_index]

                    # Make sure that we do not mess up the indexing and that everything is labeled properly
                    # If we decide to change the format we might want to adjust the font size, just to make it more readable
                    #---------------------------------------------------
                    if self.pdf_format[0] == 1:
                        current_axis = ax[col]
                    elif self.pdf_format[1] == 1:
                        current_axis = ax[row]
                    else:
                        current_axis = ax[row,col]
                    current_axis.set_xlabel(self.x_labels[pdf_index])
                    current_axis.set_ylabel(self.y_labels[pdf_index])
                    #---------------------------------------------------
                    
                    if pdf_index < n_dists:

                        current_axis.plot(x,true_npy_data[pdf_index],'b-',linewidth=self.line_width,label='True')

                        # The following has been taken from Kishans plotting tool:
                        core_value = np.mean(generated_npy_data[pdf_index],axis=0)
                        lower_limit = np.quantile(generated_npy_data[pdf_index],self.min_q,axis=0)
                        upper_limit = np.quantile(generated_npy_data[pdf_index],self.max_q,axis=0)

                        current_axis.plot(x,core_value,'r--',linewidth=self.line_width,label='GAN Prediction')
                        current_axis.fill_between(x,lower_limit,upper_limit, alpha=0.25, color="red")
                        current_axis.grid(True)
                        current_axis.legend(fontsize=self.legend_font_size)
                    
                    pdf_index += 1
                #++++++++++++++++++++++++
            #++++++++++++++++++++++++

            key_name = 'true_vs_generated_pdfs'
            plot_dict[key_name] = [fig,ax]
            plt.close()
         
        return plot_dict 
    #**********************
