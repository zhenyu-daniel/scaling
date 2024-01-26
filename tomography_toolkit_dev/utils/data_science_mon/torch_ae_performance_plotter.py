import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class AE_Performance_Plotter(object):
    """
    This class is an extension of the AE_Performance_Monitor. No complicated computation is done here. All functions defined within this class simply take numpy arrays
    as arguments and translate them into nice plots, where the plot options may be adjusted by the user
    """

    # Initialize:
    #**********************************
    def __init__(self,config={}):

        self.config = config
        
        # Basic figure layout settings:
        self.font_size = self.config['font_size'] if 'font_size' in self.config else 20 
        self.n_bins = self.config['n_bins'] if 'n_bins' in self.config else 100
        self.line_width = self.config['line_width'] if 'line_width' in self.config else 2.0
        self.fig_width_scale = self.config['fig_width_scale'] if 'fig_width_scale' in self.config else 1.0
        self.fig_height_scale = self.config['fig_height_scale'] if 'fig_height_scale' in self.config else 1.0
        self.fig_wspace = self.config['fig_wspace'] if 'fig_wspace' in self.config else 0.3
        self.legend_font_size = self.config['legend_font_size'] if 'legend_font_size' in self.config else 15

        # Residual / stat box specific settings:
        self.n_sigma_residual = self.config['n_sigma_residual'] if 'n_sigma_residual' in self.config else 5.0
        self.stat_box_x = self.config['stat_box_x'] if 'stat_box_x' in self.config else 0.05
        self.stat_box_y = self.config['stat_box_y'] if 'stat_box_y' in self.config else 0.95
        self.stat_box_font_size = self.config['stat_box_font_size'] if 'stat_box_font_size' in self.config else 15
        self.stat_box_alpha = self.config['stat_box_alpha'] if 'stat_box_alpha' in self.config else 0.5
        self.parameter_residual_format = self.config['parameter_residual_format'] if 'parameter_residual_format' in self.config else [1,1]

        # Labeling:
        self.observable_names = self.config['observable_names'] if 'observable_names' in self.config else None
        self.parameter_names = self.config['parameter_names'] if 'parameter_names' in self.config else None


        # Set the font size for the text
        plt.rcParams.update({'font.size':self.font_size})
    #**********************************

    # Check the dimension of the input arrays
    # To make things easier, all functions here assume that the input arrays are 2D
    # So 1D arrays need to be simply reshaped from: (N,) to: (N,1)
    # We assume that the event axis is 0 and the feature axis is 1, i.e. every row in the array represents an event
    #**********************************
    def check_data_dimensionality(self,npy_data,feature_axis=1):
        if len(npy_data.shape) == 1:
            return np.expand_dims(npy_data,axis=feature_axis)
        return npy_data
    #**********************************

    # Add a stats box:
    # The code from this is mainly taken from here: https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
    # We use the scipy.stats.descibe function to retreive the relevant information from a given distribution:
    #**********************************
    def get_stats(self,distribution,plot_axes,x_box,y_box,box_font_size,box_alpha):
        # Collect the main stats by conveniently using the describe function:
        n_entries,range,mean,var,kurt,skew = describe(distribution,axis=0)

        # Add the stats information to the plot, if a plot axis is provided:
        if plot_axes is not None:
           stats = '\n'.join((
            'Entries: %.0f' % (n_entries),
            'Mean:  %.3f' % (mean),
            'Variance: %.3f' % (var),
            'Kurtosis: %.3f' % (kurt),
            'Skewness: %.3f' % (skew)
           ))

           box = dict(boxstyle='round', facecolor='wheat', alpha=box_alpha)
           plot_axes.text(x_box,y_box,stats, transform=plot_axes.transAxes,fontsize=box_font_size,
               verticalalignment='top', bbox=box)

        return n_entries,range,mean,var,kurt,skew
    #**********************************

    # Plot losses and accuracy:

    #**********************************
    def plot_loss_and_dropout_rate(self,gen_loss,dropout_rate,x_label="Training Epoch"):
        # Check the dimensionality:
        gen_loss = self.check_data_dimensionality(gen_loss)
        dropout_rate = self.check_data_dimensionality(dropout_rate)

        fig_width = int(self.fig_width_scale * 15)
        fig_height = int(self.fig_height_scale * 8)

        fig,ax = plt.subplots(1,2,figsize=(fig_width,fig_height),sharex=True)
        fig.suptitle('AE Training Results')
        fig.subplots_adjust(wspace=self.fig_wspace)

        x_values = [i for i in range(1,1+len(gen_loss))]

        # Plot the loss first:
        ax[0].plot(x_values,gen_loss,'b-',linewidth=self.line_width)
        ax[0].grid(True)
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Encoder Training Loss')

        # Now check dropout rate:
        ax[1].plot(x_values,dropout_rate*100.0,'k-',linewidth=self.line_width)
        ax[1].grid(True)
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel('MC Dropout Rate [%]')
        ax[1].set_ylim(0.0,100.0)

        plot_dict = {'loss_and_dropout_rate_plots': [fig,ax]}

        plt.close()
        return plot_dict
    #**********************************

    # Compare predicted and generated data and residuals:

    #**********************************
    def plot_observables_and_residuals(self,real_data,generated_data,residuals):
        # Check dimensionality:
        real_data = self.check_data_dimensionality(real_data)
        generated_data = self.check_data_dimensionality(generated_data)
        residuals = self.check_data_dimensionality(residuals)
        
        # Get dimension of observables:
        feat_dim = residuals.shape[1]

        fig_width = int(self.fig_width_scale * 15)
        fig_height = int(self.fig_height_scale * 8)

        plot_dict = {}
        # Compare real and generated data:
        #++++++++++++++++++++++++++++
        for i in range(feat_dim):
            f = i + 1
            if self.observable_names is not None:
                f = self.observable_names[i]


            fig,ax = plt.subplots(1,2,figsize=(fig_width,fig_height),sharey=True)
            fig.suptitle('Comparing real and generated Data')

            # Look at distributions directly:
            _,obs_range,_,_,_,_ = self.get_stats(real_data[:,i],None,None,None,None,None)
            min_obs_x = obs_range[0]
            max_obs_x = obs_range[1]

            ax[0].hist(real_data[:,i],self.n_bins,histtype='step',color='k',linewidth=self.line_width,label='Real',range=[min_obs_x,max_obs_x])
            ax[0].hist(generated_data[:,i],self.n_bins,histtype='step',color='r',linewidth=self.line_width,label='Generated',range=[min_obs_x,max_obs_x])
            ax[0].grid(True)
            ax[0].legend(fontsize=self.legend_font_size)
            ax[0].set_xlabel(str(f))
            ax[0].set_ylabel('Entries')

            # Check residuals:
            # Collect basic information, using scipy.stats.describe:
            _,_,res_mean,res_var,_,_ = self.get_stats(residuals[:,i],ax[1],self.stat_box_x,self.stat_box_y,self.stat_box_font_size,self.stat_box_alpha)

            res_min = res_mean - self.n_sigma_residual*np.sqrt(res_var)
            res_max = res_mean + self.n_sigma_residual*np.sqrt(res_var)
            ax[1].hist(residuals[:,i],self.n_bins,color='k',histtype='step',linewidth=self.line_width,range=[res_min,res_max])


            ax[1].set_xlabel('Residual ' + str(f))
            ax[1].grid(True)

            key_name = 'real_vs_gen_data_plots_' + str(f)
            plot_dict[key_name] = [fig,ax]

            plt.close()

            # Check, if the correlations are modeled properly:
            #++++++++++++++++++++++++++++
            for j in range(feat_dim):
                if j > i:
                    g = j + 1
                    if self.observable_names is not None:
                       g = self.observable_names[j]

                    plot_range = [
                        [np.min(real_data[:,i]),np.max(real_data[:,i])],
                        [np.min(real_data[:,j]),np.max(real_data[:,j])],
                    ]

                    cfig,cax = plt.subplots(1,2,figsize=(fig_width,fig_height),sharey=True)
                    title = 'Correlation between ' + str(f) + ' and ' + str(g)
                    cfig.suptitle(title)

                    cax[0].set_title('Real Data')
                    cax[0].hist2d(real_data[:,i],real_data[:,j],self.n_bins,norm=LogNorm(),range=plot_range)
                    cax[0].grid(True)
                    cax[0].set_xlabel(str(f))
                    cax[0].set_ylabel(str(g))

                    cax[1].set_title('Generated Data')
                    cax[1].hist2d(generated_data[:,i],generated_data[:,j],self.n_bins,norm=LogNorm(),range=plot_range)
                    cax[1].grid(True)
                    cax[1].set_xlabel(str(f))

                    key_name = 'data_correlation_plots_' + str(g) + '_vs_' + str(f)
                    plot_dict[key_name] = [cfig,cax]

                    plt.close()
            #++++++++++++++++++++++++++++

        #++++++++++++++++++++++++++++

        return plot_dict
    #**********************************

    # Check residuals between predicted and true parameters:

    #**********************************
    def plot_parameter_residuals(self,true_parameters,pred_parameters):
        # Calculate the parameter residuals:
        par_residuals = true_parameters - pred_parameters
        par_residuals = self.check_data_dimensionality(par_residuals)
        n_features = par_residuals.shape[1]

        # Set dimensions for plots:
        rfig_width = int(self.fig_width_scale *self.parameter_residual_format[1] * 2 + 10)
        rfig_height = int(self.fig_height_scale * self.parameter_residual_format[0] * 2 + 6)
        
        # Set up a plotting dictionary:
        plot_dict = {}

        # Depending on the set format, we may either create n individual parameter plots (this might be helpful if you have >= 20 parameters)
        # or if you prefer to show the parameter residuals all in one plot (helpful for a small number of paraemeters)

        # TODO: The following implementation is not pretty and needs to be optimized in the near future.

        if self.parameter_residual_format[0] == 1 and self.parameter_residual_format[1] == 1:

            #+++++++++++++++++++++++++++++++++++
            for k in range(n_features):
               f = k +1
               if self.parameter_names is not None:
                  f = self.parameter_names[k]

            
               rfig,rax = plt.subplots(figsize=(rfig_width,rfig_height))

               # Check residuals:
               # Collect basic information, using scipy.stats.describe:
               _,_,res_mean,res_var,_,_ = self.get_stats(par_residuals[:,k],rax,self.stat_box_x,self.stat_box_y,self.stat_box_font_size,self.stat_box_alpha)

               res_min = res_mean - self.n_sigma_residual*np.sqrt(res_var)
               res_max = res_mean + self.n_sigma_residual*np.sqrt(res_var)

               rax.hist(par_residuals[:,k],self.n_bins,color='k',histtype='step',linewidth=self.line_width,range=[res_min,res_max])
               rax.grid(True)
               rax.set_xlabel('Residual Parameter ' + str(f))
               rax.set_ylabel('Entries')

               key_name = 'parameter_residual_plots_' + str(f)
               plot_dict[key_name] = [rfig,rax]

               plt.close()
           #+++++++++++++++++++++++++++++++++++

        else:
            rfig,rax = plt.subplots(self.parameter_residual_format[0],self.parameter_residual_format[1],figsize=(rfig_width,rfig_height),sharey=True)
            rfig.subplots_adjust(wspace=self.fig_wspace)

            par_index = 0
            current_axis = None
            #++++++++++++++++++++++++
            for row in range(self.parameter_residual_format[0]):
                #++++++++++++++++++++++++
                for col in range(self.parameter_residual_format[1]):
                    f = par_index +1
                    if self.parameter_names is not None:
                          f = self.parameter_names[par_index]


                    # Make sure that we do not mess up the indexing and that everything is labeled properly
                    # If we decide to change the format we might want to adjust the font size, just to make it more readable
                    #---------------------------------------------------
                    if self.parameter_residual_format[0] == 1:
                        current_axis = rax[col]
                        if col == 0:
                            current_axis.set_ylabel('Entries',fontsize=15)
                        current_axis.set_xlabel('Residual Parameter ' + str(f),fontsize=15)
                    elif self.parameter_residual_format[1] == 1:
                        current_axis = rax[row]
                        current_axis.set_xlabel('Residual Parameter ' + str(f),fontsize=15)
                        current_axis.set_ylabel('Entries',fontsize=15)
                    else:
                        current_axis = rax[row,col]
                        if col == 0:
                            current_axis.set_ylabel('Entries',fontsize=15)
                        current_axis.set_xlabel('Residual Parameter ' + str(f),fontsize=15)

                    current_axis.tick_params(axis='x',labelsize=15)
                    current_axis.tick_params(axis='y',labelsize=15)
                    #---------------------------------------------------

                    if par_index < n_features: #--> Just make sure that you dont accidentally break anything:
                       
                       # Check residuals:
                       # Collect basic information, using scipy.stats.describe:
                       _,_,res_mean,res_var,_,_ = self.get_stats(par_residuals[:,par_index],current_axis,self.stat_box_x,self.stat_box_y,10,0.1)

                       res_min = res_mean - self.n_sigma_residual*np.sqrt(res_var)
                       res_max = res_mean + self.n_sigma_residual*np.sqrt(res_var)

                       current_axis.hist(par_residuals[:,par_index],self.n_bins,color='k',histtype='step',linewidth=self.line_width,range=[res_min,res_max])
                       current_axis.grid(True)
                       
                    par_index += 1 
                #++++++++++++++++++++++++
            #++++++++++++++++++++++++

            plot_dict['parameter_residual_plots'] = [rfig,rax]
            plt.close()

        # Now produce a nice plot:
        core_val = np.mean(par_residuals / true_parameters,0) * 100.0
        core_err = np.std(par_residuals / true_parameters,0) * 100.0
        pars = np.array([j for j in range(1,1+n_features)])

        ffig_width = int(self.fig_width_scale * 10)
        ffig_height = int(self.fig_height_scale * 8)

        figf,axf = plt.subplots(figsize=(ffig_width,ffig_height))
        figf.suptitle('Gan Parameter Prediction')

        axf.errorbar(x=pars,y=core_val,yerr=core_err,fmt='ko',linewidth=self.line_width,markersize=10)
        axf.plot([pars[0],pars[n_features-1]],[0.0,0.0],'r--',linewidth=self.line_width)
        axf.set_xticks(pars)
        if self.parameter_names is not None:
            axf.set_xticklabels(self.parameter_names)
        axf.set_xlabel('Parameter')
        axf.set_ylabel('Avg. (True - GAN) / True [%]')
        axf.grid(True)

        plot_dict['parameter_comparison_plots'] = [figf,axf]

        # Create seaborn plot of correlations
        #++++++++++++++++++++++++++++
        try:
            import pandas as pd
            import seaborn as sns
            # create pairwise correlation plot
            column_labels = ['param_{}'.format(i+1) for i in range(n_features)]
            pred_parameters_df = pd.DataFrame(data=pred_parameters, columns=column_labels)
            g = sns.pairplot(pred_parameters_df, corner=False,
                             kind='scatter', plot_kws={'alpha': 0.5},
                             diag_kind='kde' #, diag_kws={'bins':20, 'binrange':(0.0,1.0)}
            )
            # format pairwise correlation plot
            for i in range(n_features):
                for j in range(n_features):
                    g.axes[i,j].set_xlim((0.0, 1.0))
                    g.axes[i,j].grid()
                    if i != j:
                        g.axes[i,j].set_ylim((0.0, 1.0))
            g.map_offdiag(sns.kdeplot, levels=4, color='purple')
            g.map_offdiag(sns.regplot, scatter=False, color='purple', line_kws={'linestyle':'dashed'})
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        g.axes[i,j].axvline(x=true_parameters[i], color='green')
                    else:
                        g.axes[i,j].scatter(true_parameters[j], true_parameters[i], color='darkgreen',
                                            s=16**2, marker='+', linewidth=4)

            plot_dict['parameter_correlation_plots'] = [g.fig,None]
        except ImportError:
            plot_dict['parameter_correlation_plots'] = None
            pass

        plt.close()
        return plot_dict
    #**********************************
