#import unittest2 as unittest
import unittest
import torch
import tomography_toolkit_dev.eventselection_module as eventselection_agent


class RectangularEventSelectionTests(unittest.TestCase):
      
    """
    This test runs as follows: Create a data set with N features
    Define a set of limits / cuts for each feature
    Apply cuts on data set and check if the new data set is within the limits that just have been defined
    """

    # Test the blank
    def test_blank(self):
        # Create data set with 3 features
        n_evts = 1000000
        data = torch.cat([
            torch.normal(mean=0.5,std=0.2,size=(n_evts,1)),
            torch.normal(mean=-1.5,std=0.8,size=(n_evts,1)),
            torch.normal(mean=3.0,std=1.0,size=(n_evts,1))
        ],dim=1)

        # Load the event selection module
        conf = {}
        evtsel_module = eventselection_agent.make("rectangular_cuts",config=conf)

        # Filter the data. Since no limits have been specified,
        # the filter simply returns the original data
        filtered_data = evtsel_module.apply(data)

        # Ideally, the sume of the difference between the two data sets should be zero:
        diff = filtered_data - data
        sum_diff = torch.sum(diff)

        self.assertEqual(sum_diff,0.0)

    # Test the hard cut flow which is pretty straight forward
    def test_hard_cut_flow(self):
        # Create data set with 3 features
        n_evts = 1000000
        data = torch.cat([
            torch.normal(mean=0.5,std=0.2,size=(n_evts,1)),
            torch.normal(mean=-1.5,std=0.8,size=(n_evts,1)),
            torch.normal(mean=3.0,std=1.0,size=(n_evts,1))
        ],dim=1)

        # Specify limits for a few features:
        f1_min = 0.0
        f1_max = 0.7

        f2_min = -1E9
        f2_max = 1E9

        f3_min = -1E9
        f3_max = 3.5

        min_lim = [f1_min,f2_min,f3_min]
        max_lim = [f1_max,f2_max,f3_max]

        conf = {
            'rect_evt_filter_minimum': min_lim,
            'rect_evt_filter_maximum': max_lim
        }
        
        # Load the event selection module
        evtsel_module = eventselection_agent.make("rectangular_cuts",config=conf)

        # Filter the data:
        filtered_data = evtsel_module.apply(data) # Use .apply() for the hard cut flow (i.e. outside any training loop)

        # Now determine the limits of the filtered data set:
        min_filterd_data, _ = torch.min(filtered_data,dim=0)
        max_filterd_data, _ = torch.max(filtered_data,dim=0)

        # And make sure that the filtered data is within the previosuly defined limits:
        limits = (min_filterd_data[0] >= f1_min) & (max_filterd_data[0] <= f1_max) & (max_filterd_data[2] <= f3_max)
        limits = limits.numpy()

        # If everything makes sense, the limit check should return true:
        self.assertTrue(limits,True)   

    # Test the soft cut flow which requires some thinking
    def test_soft_cut_flow(self):
        # Create data set with 3 features
        n_evts = 1000000
        data = torch.cat([
            torch.normal(mean=0.5,std=0.2,size=(n_evts,1)),
            torch.normal(mean=-1.5,std=0.8,size=(n_evts,1)),
            torch.normal(mean=3.0,std=1.0,size=(n_evts,1))
        ],dim=1)

        # Specify limits for a few features:
        f1_min = 0.0
        f1_max = 0.7

        f2_min = -1E9
        f2_max = 1E9

        f3_min = -1E9
        f3_max = 3.5

        min_lim = [f1_min,f2_min,f3_min]
        max_lim = [f1_max,f2_max,f3_max]
        
        conf = {
            'rect_evt_filter_minimum': min_lim,
            'rect_evt_filter_maximum': max_lim
        }
        
        # Load the event selection module
        evtsel_module = eventselection_agent.make("rectangular_cuts",config=conf)

        # Filter the data:
        filtered_data = evtsel_module.forward(data) # Use .forward() for the soft cut, i.e. inside the training loop

        # Unlike the test above, we can not use the limits of the new data directly, because we are using a 'soft' cut
        # However, we know: X_filtered = f * X, where f = 1, if X is within the limits. Thus, we expect that 
        # All events in X_filtered, that are exactly equal to X, should be within the specified limits

        # Look for events that are exact equal
        equal = filtered_data == data 
        f1_data = filtered_data[:,0][equal[:,0]] 
        f3_data = filtered_data[:,2][equal[:,2]]

        # Determine limits for events that are exact equal:
        obs_f1_min,_ = torch.min(f1_data,dim=0)
        obs_f1_max,_ = torch.max(f1_data,dim=0)
        obs_f3_max,_ = torch.max(f3_data,dim=0)
 
        # Check that new limits are within the specified ones:
        limits = (obs_f1_min > f1_min) & (obs_f1_max < f1_max) & (obs_f3_max < f3_max)

        # If everything makes sense, the limit check should return true:
        self.assertTrue(limits,True)   


if __name__ == "__main__":
    unittest.main()
