import unittest
#import unittest2
import tomography_toolkit_dev.experimental_module as experiments
import tomography_toolkit_dev.eventselection_module as filters
import tomography_toolkit_dev.expdata_module as data_parsers

import numpy as np
import torch
import os

class TestExpDataModule(unittest.TestCase):

    # Create and load a test data set
    #*********************************
    def create_test_data(self,n_evts):
        # Create a simple data set with format: N_features x N_events
        data = np.concatenate([
            np.random.normal(loc=-1.0,scale=0.5,size=(1,n_evts)),
            np.random.normal(loc=0.0,scale=1.0,size=(1,n_evts)),
            np.random.normal(loc=1.0,scale=0.7,size=(1,n_evts))
        ],axis=0)
    
        data_t = (data,np.array(3.142),np.array(42.0),np.array(11.11))
        test_data = np.asarray(data_t,dtype=object)

        np.save('test_data.npy',test_data)#--> We create the test data this way, because we want to check
        # the entire functionality of the exp data module, which includes loading a .npy file
    #*********************************

    # Run the test data through the 
    # experimental data module
    #*********************************
    def run_parser(self,n_events,data_config,experimental_config,is_training):
        # Create the data:
        self.create_test_data(n_events)  

        # Load modules:
        # Experiment:
        experiment = experiments.make("simple_det",config=experimental_config)
        # Event selection:
        filter = filters.make("rectangular_cuts",config=experimental_config)
        # Set up the module chain:
        module_list = [None,experiment,filter,None]
        # We do not use a theory module or discriminator module here
        # so we just replcae those by 'None'

        # Load the data parser:
        parser = data_parsers.make('numpy_data_parser',config=data_config,module_chain=module_list) 

        # Get the data package:
        data = parser.return_data(is_training=is_training,additional_observable_names=['var1','var2','var3'])

        # Remove the test data:
        os.remove('test_data.npy')
        return data
    #*********************************
    
    # First, basic test:
    #*********************************
    def test_data_retrieval(self):
        n_events = 1000

        data_cfg = {
            "path": "test_data.npy"
        }

        experimental_cfg = {
            "smearing_parameters": [0.1,0.05,0.08],
            "correlation_parameters": [-0.02,0.07],
            "rect_evt_filter_minimum": [-1.5,-0.5,0.5],
            "rect_evt_filter_maximum": [1.5,0.5,1.5]
        }

        # Get the experimental and original data:
        data = self.run_parser(n_events,data_cfg,experimental_cfg,False)
        
        exp_data = data['parsed_data']
        orig_data = data['original_data']

        # Since there are no experimental and event selection module are involved,
        # this data is the original data. So we check that the experimental and original data set are equal:
        first_test = False
        sum_diff = torch.sum(exp_data - orig_data).numpy()
        if sum_diff == 0.0:
            first_test = True

 
        # Now we create another data set, but include the transpose transformation:
        data_cfg['transpose_dim'] = [0,1]

        # Get the 'new' data set:
        data_new = self.run_parser(n_events,data_cfg,experimental_cfg,False)
        exp_data_new = data_new['parsed_data']
        orig_data = data_new['original_data']

        # Run a second test. Assuming that the first test passed, we can conclude that
        # This data set is also equal to the original, but with transposed dimensions:
        second_test = False
        if exp_data_new.size()[0] == orig_data.size()[0] and exp_data_new.size()[1] == orig_data.size()[1]:
            second_test = True

        # Lastly, check that the remaining data (here: 3 floats) are equal to the ones specified above:
        third_test = False 

        diff = (data_new['var1'].numpy() - 3.142) + (data_new['var2'].numpy() - 42.0) + (data_new['var3'].numpy() - 11.11)
        diff = round(diff*diff,3)

        if diff == 0.0:
            third_test = True
        
        self.assertTrue(first_test & second_test & third_test, True)
    #*********************************

    # Run a second test, using the 
    # experimental model only
    #*********************************
    def test_data_after_exp_module(self):
        n_events = 1000

        data_cfg = {
            "path": "test_data.npy",
            "use_exp_module": True,
            "use_evtsel_module": False,
            "transpose_dim": [0,1] #--> We need the transpose function here, because the experimental module
            # 'prefers' the format: N_events x N_features
        }

        experimental_cfg = {
            "smearing_parameters": [0.1,0.05,0.08],
            "correlation_parameters": [-0.02,0.07],
            "rect_evt_filter_minimum": [-1.5,-0.5,0.5],
            "rect_evt_filter_maximum": [1.5,0.5,1.5]
        }

        # Get the experimental data:
        data = self.run_parser(n_events,data_cfg,experimental_cfg,False)
        exp_data = data['parsed_data']
        orig_data = data['original_data']

        # This time, both data sets should be different, due to the implemented
        # resolution effects. We do not need to test this experimental module - This is doen in 
        # a separate unittest
        first_test = False
        sum_diff = torch.sum(exp_data - orig_data).numpy()
        if sum_diff != 0.0:
            first_test = True

        # Repeat the exact same test, but under the 'training' condition. For the detector module used here,
        # the training and non-training response is exactly equal:
        data_new = self.run_parser(n_events,data_cfg,experimental_cfg,True)
        exp_data_new = data_new['parsed_data']
        orig_data = data_new['original_data']

        second_test = False
        sum_diff = torch.sum(exp_data_new - orig_data).numpy()
        if sum_diff != 0.0:
            second_test = True

        self.assertTrue(first_test & second_test, True)
    #*********************************

    # Run a last test, with the event selection module only:
    #*********************************
    def test_data_after_evtsel_module(self):
        n_events = 1000

        data_cfg = {
            "path": "test_data.npy",
            "use_exp_module": False,
            "use_evtsel_module": True,
            "transpose_dim": [0,1] #--> We need the transpose function here, because the experimental module
            # 'prefers' the format: N_events x N_features
        }

        experimental_cfg = {
            "smearing_parameters": [0.1,0.05,0.08],
            "correlation_parameters": [-0.02,0.07],
            "rect_evt_filter_minimum": [-1.5,-0.5,0.5],
            "rect_evt_filter_maximum": [1.5,0.5,1.5]
        }

        # Get the experimental data, this time including the event selection module:
        data = self.run_parser(n_events,data_cfg,experimental_cfg,False)
        exp_data = data['parsed_data']

        # Since we use the hard cut option here, we can check the limits of the exp data set directly:
        obs_min,_  = torch.min(exp_data,dim=0)
        obs_max,_ = torch.max(exp_data,dim=0)

        first_test = False
        min_pass = obs_min[0] >= -1.5 and obs_min[1] >= -0.5 and obs_min[2] >= 0.5
        max_pass = obs_max[0] <= 1.5 and obs_max[1] <= 0.5 and obs_max[2] <= 1.5

        if min_pass & max_pass:
            first_test = True

        # Now we run the entire chain under the 'training' condition, i.e. the soft cut is
        # applied to the data
        data_new = self.run_parser(n_events,data_cfg,experimental_cfg,True)
        exp_data_new = data_new['parsed_data']
        orig_data = data_new['original_data']

        # There is not much we can test here, except that the exp data and the original
        # data have the same dimension, but different values (due to the nature of the soft cut)
        second_test = False
        sum_diff = torch.sum(exp_data_new - orig_data).numpy()
        if sum_diff != 0.0:
            second_test = True

        # Lastly, we run again under the training condition, but this time we force the event selection module
        # to disable its behavior during training, i.e. the .forward() / .apply() function do the exact same:
        experimental_cfg = {
            "smearing_parameters": [0.0,0.0,0.0],
           # "smearing_parameters": [0.1,0.05,0.08],
           # "correlation_parameters": [-0.02,0.07],
            "rect_evt_filter_minimum": [-1.5,-0.5,0.5],
            "rect_evt_filter_maximum": [1.5,0.5,1.5],
            "rect_evt_filter_force_hard_cut_in_training": True
        }

        data_new_new = self.run_parser(n_events,data_cfg,experimental_cfg,True)
        exp_data_new_new = data_new_new['parsed_data']
        
        # Since we force the filter module to do the exact same during training and analysis,
        # we can simply check that our filtered data is exactly between the new limits
        third_test = False
        obs_min_new_new,_  = torch.min(exp_data_new_new,dim=0)
        obs_max_new_new,_ = torch.max(exp_data_new_new,dim=0)

        min_pass = obs_min_new_new[0] >= -1.5 and obs_min_new_new[1] >= -0.5 and obs_min_new_new[2] >= 0.5
        max_pass = obs_max_new_new[0] <= 1.5 and obs_max_new_new[1] <= 0.5 and obs_max_new_new[2] <= 1.5

        if min_pass & max_pass:
            third_test = True

        self.assertTrue(first_test & second_test & third_test, True)
    #*********************************


if __name__ == "__main__":
    # You can run this file using:   python unittest_example.py
    unittest.main()
