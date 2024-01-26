#import unittest2 as unittest
import unittest
import torch
import tomography_toolkit_dev.experimental_module as experimental_agent


class SimpleDetectorTests(unittest.TestCase):
    # Test ideal detector first:
    #***********************************************************
    def test_ideal_detector(self):
        conf = {'smearing_parameters':[0.0,0.0]}

        # Get ideal detector:
        ideal_det = experimental_agent.make("simple_det",config=conf)
        
        # Define a simple enough data set:
        in_data = torch.normal(mean=0.5,std=0.2,size=(100,2))

        # And pass it through the identity matrix:
        out_data = ideal_det.forward(in_data)

        # Calculate residuals which have to be exact zero:
        residuals = in_data - out_data
        sum_residuals = torch.sum(residuals)
        
        self.assertEqual(sum_residuals,0)
    #***********************************************************

    # Test detector with relative gaussian smearing, NO correlations and a specified event axis
    #***********************************************************

    # Run a chi^2 hypothesis test (suggested by Malachi)
    def run_chiq_test(self,smear,in_data,evt_axis):
        # Get the number of features
        n_f = len(smear) #--> Number of features
        
        conf = {'smearing_parameters':smear}
        transpose = None # Do not transpose vector, if event axis is 0
        if evt_axis == 1:
          conf['transpose_dim'] = (1,0)

        # Load gaussian detector:
        gauss_det = experimental_agent.make("simple_det",config=conf)

        # Pass data through detector:
        out_data = gauss_det.forward(in_data)

        # Retreive the mean and sigma from the incoming, gaussian, data:
        m_in = torch.mean(in_data,dim=evt_axis)
        s_in = torch.std(in_data,dim=evt_axis)

        # Using gaussian smearing on a gaussian data set is basically multiplying two gaussian distributions
        # Thus, we need to know the mean and sigma of the folding distribution which are simply:
        m_fold = torch.tensor(n_f*[1.0])
        s_fold = torch.tensor(smear)

        # Now we can calculate the expected mean and sigma of the outcoming data:
        m_exp = m_in * m_fold
        s_exp = torch.sqrt( (m_in**2 + s_in**2) * (m_fold**2 + s_fold**2) - m_exp**2 )
        
        # Finally , we can calculate the chi^2 per degree of freedom:
        arg = (out_data - m_exp) / s_exp
        test_chisq = torch.sum(arg**2) / (n_f*in_data.size()[evt_axis])
        test_chisq = test_chisq.numpy()

        # Test, if the resulting chi^2 per ndf is smaller than 2.5 (ideally, one would expect 1)
        self.assertLessEqual(test_chisq,2.5)
    
    #-------------------------------------------

    # Run the chi^2 test for a specific data set    
    def test_gaussian_detector_1(self):
        # Define incoming data:
        in_data = torch.normal(mean=0.5,std=0.2,size=(100000,4))

        # Smearing parameters: (one for each feature dimension)
        smear_p = [0.07,0.1,0.5,0.2]

        self.run_chiq_test(smear_p,in_data,0)

    #-------------------------------------------

    # Run test again, but this time with a different event axis:
    def test_gaussian_detector_2(self):
        # Define incoming data:
        in_data = torch.normal(mean=0.5,std=0.2,size=(4,100000))

        # Smearing parameters: (one for each feature dimension)
        smear_p = [0.07,0.1,0.5,0.2]

        self.run_chiq_test(smear_p,in_data,1)
    #***********************************************************

    # Important note: unit-test for a smearing with data point correlations
    # is missing.



if __name__ == "__main__":
    # You can run this file using:   python unittest_example.py
    unittest.main()
