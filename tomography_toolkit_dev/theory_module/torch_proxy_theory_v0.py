import numpy as np
import matplotlib.pyplot as plt
import torch
import functorch
from tomography_toolkit_dev.core.theory_core import Theory

class proxy_theory(Theory):
    """
    
    """

    def __init__(self, config, devices="cpu"):
        self.devices = devices
        self.nParameters = config['n_parameters'] if 'n_parameters' in config else 6
        self.parmin = config['parmin'] if 'parmin' in config else 0
        self.parmax = config['parmax'] if 'parmax' in config else 1
        self.name = "Torch Proxy Theory"
        self.disable_training_behavior = False #--> This flag is only needed when we wish that .forward() diverges from its behavior 
        # during training. We basically force that .forward() does the same as .apply()
        # This flag is not used here, but there might be cases where we have to have this option

        if isinstance(self.parmin, int):
            self.parmin = [self.parmin for i in range(self.nParameters)]
            self.parmin = np.array(self.parmin)
        if isinstance(self.parmax, int):
            self.parmax = [self.parmax for i in range(self.nParameters)]
            self.parmax = np.array(self.parmax)
        self.parmin = torch.as_tensor(self.parmin,device=self.devices)
        self.parmax = torch.as_tensor(self.parmax,device=self.devices)

        self.xmin, self.xmax = 0.1, 0.99999
        self.dx = (self.xmax - self.xmin) / 1000
        self.x_full_range = torch.arange(self.xmin, self.xmax, self.dx,device=self.devices)
        
        

    def get_ud(self, p):
        u = p[0]*torch.pow(self.x_full_range, p[1]) * torch.pow((1-self.x_full_range), p[2])
        d = p[3]*torch.pow(self.x_full_range, p[4]) * torch.pow((1-self.x_full_range), p[5])
        return u, d

    def integral_approx(self, y):
            return torch.sum(y[:-1]) * self.dx

    def torch_interp(self, x, xs, ys):
        # determine the output data type
        ys = torch.Tensor(ys)
        dtype = ys.dtype

        # normalize data types
        # ys = ys.type(torch.float64).to(self.devices)
        # xs = xs.type(torch.float64).to(self.devices)
        # x = x.type(torch.float64).to(self.devices)

        # We need to use float32 for the macbook mps...
        ys = ys.type(torch.float32).to(self.devices)
        xs = xs.type(torch.float32).to(self.devices)
        x = x.type(torch.float32).to(self.devices)


        # pad control points for extrapolation
        xs = torch.cat([torch.tensor([torch.finfo(xs.dtype).min]).to(self.devices)
                        , xs, torch.tensor([torch.finfo(xs.dtype).max]).to(self.devices)
                       ], axis=0)
        ys = torch.cat([ys[:1], ys, ys[-1:]], axis=0)

        # compute slopes, pad at the edges to flatten
        ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        ms = torch.nn.functional.pad(ms[:-1], (1, 1))

        # solve for intercepts
        bs = ys - ms*xs

        # search for the line parameters at each input data point
        # create a grid of the inputs and piece breakpoints for thresholding
        # rely on argmax stopping on the first true when there are duplicates,
        # which gives us an index into the parameter vectors
        
        #Argmax not implemented for boolean on CPU
        double_from_bool = xs[..., None, :] > x[..., None]
        #double_from_bool = double_from_bool.double()
        double_from_bool = double_from_bool.float() # --> Need to use float for macbook mps...
        i = torch.argmax(double_from_bool, dim=-1)
        m = ms[..., i]
        b = bs[..., i]

        # apply the linear mapping at each input data point
        y = m*x + b
        
        return torch.reshape(y, x.shape).type(dtype)

    def inverse_cdf(self, cdf_allx1, nevents):
        u = torch.rand((1*nevents,))*0.9999
        cdf_sort_indx1 = torch.argsort(cdf_allx1)
        cdf_sort1 = cdf_allx1[cdf_sort_indx1]
        x_sort1 = self.x_full_range[cdf_sort_indx1]
        
        events_out1 = self.torch_interp(u, cdf_sort1, x_sort1)
        events_out1 = torch.reshape(events_out1, (nevents,))
        return events_out1

    def gen_events(self, true_params, nevents):
        # Denormalize the parameters
        true_params = true_params * (self.parmax - self.parmin) + self.parmin

        u_full, d_full = self.get_ud(true_params)
        sigma1 = 4*u_full+d_full
        sigma2 = 4*d_full+u_full
        
        norm1 = self.integral_approx(sigma1)
        norm2 = self.integral_approx(sigma2)

        u, d = self.get_ud(true_params)
        
        
        pdf1 = ((4*u)+d)/norm1
        pdf2 = ((4*d)+u)/norm2
        
        inv_indices = torch.arange(pdf1.size(0)-1, -1, -1,device=self.devices).long()
        inv_pdf1 = pdf1.index_select(0, inv_indices)
        
        inv_indices = torch.arange(pdf2.size(0)-1, -1, -1,device=self.devices).long()
        inv_pdf2 = pdf2.index_select(0, inv_indices)
        
        inv_cdf_allx1 = (torch.cumsum(inv_pdf1*self.dx, dim=0)/torch.sum(pdf1*self.dx))
        inv_cdf_allx2 = (torch.cumsum(inv_pdf2*self.dx, dim=0)/torch.sum(pdf2*self.dx))
        
        indices = torch.arange(inv_cdf_allx1.size(0)-1, -1, -1,device=self.devices).long()
        cdf_allx1 = inv_cdf_allx1.index_select(0, indices)
        
        indices = torch.arange(inv_cdf_allx2.size(0)-1, -1, -1,device=self.devices).long()
        cdf_allx2 = inv_cdf_allx2.index_select(0, indices)
        
        events1 = self.inverse_cdf(cdf_allx1, nevents)
        events2 = self.inverse_cdf(cdf_allx2, nevents)

        events = torch.cat([torch.unsqueeze(events1, 0), torch.unsqueeze(events2, 0)], dim=0).to(self.devices)
       
        return events, norm1, norm2

    def paramsToEventsMap(self, params, nevents):
        return functorch.vmap(lambda x:self.gen_events(x, nevents), in_dims=0, randomness="different")(params)

    def forward(self, params, nevents=1):
        return self.paramsToEventsMap(params, nevents)
        
    def apply(self, params, nevents=1):
        return self.paramsToEventsMap(params, nevents)
        
    
