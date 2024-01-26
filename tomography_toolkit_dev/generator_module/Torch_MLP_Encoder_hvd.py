import torch
import numpy as np
from torch.nn import Linear, LeakyReLU, ReLU, Tanh, Sigmoid, Dropout, Sequential, BatchNorm1d
from collections import OrderedDict
from tomography_toolkit_dev.core.generator_core import Generator

def _custom_weights_init(m):
    ''' Custom weights initialization '''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

class Torch_MLP_Encoder(torch.nn.Module, Generator):
    """
    A class to define multi-layer perceptron based generator for the GAN workflow.
    Inherits from torch Module and Generator core classes.
    ...
    Attributes
    ----------
    config: Dict
        A dictionary containing following parameters:
        - key: 'num_layers'
            val_type: int,
            val: number of hidden layers for the model
        - key: 'num_nodes'
            val_type: int or list/array of ints,
            val: number of nodes in each hidden layer if same specify a single integer else an array of length equal to config['num_layers'] with each entry specifying number of nodes in that layer
        - key: 'dropout_percents'
            val_type: float or list of floats
            val: dropout percentage for each dropout layer
        - key: 'input_size'
            val_type: int,
            val: size of the input noise
        - key: 'output_size'
            val_type: int
            val: size of the output layer / number of outputs

    module_chain: list
        A list of modules that comes after generator_module all the way to and including discriminator_module


    Methods
    -------
    forward(inputs)
        Forward pass of the inputs to produce output

    generate(noise)
        Generates and returns n events where n is defined by dimension of noise input

    train(noise)
        Performs backpropagation and training of generator model


    """
    def __init__(self, config, module_chain, devices="cpu"):
        super(Torch_MLP_Encoder, self).__init__()
        self.devices = devices
        self.nLayers = int(config['num_layers']) if 'num_layers' in config else 4
        self.nNodes = config['num_nodes'] if 'num_nodes' in config else self.nLayers*[128]
        self.activations = config['activation'] if 'activation' in config else self.nLayers*["LeakyRelu"]
        self.useBias = config['use_bias'] if 'use_bias' in config else self.nLayers*[True]
        self.batchNorms = config['batchNorms'] if 'batchNorms' in config else self.nLayers*[False]
        self.dropouts = config['dropout_percents'] if 'dropout_percents' in config else self.nLayers*[0.]
        self.applyCustomWeightsInit = config['applyCustomWeightsInit'] if 'applyCustomWeightsInit' in config else False
        self.inputSize = config['input_size'] if 'input_size' in config else 10
        self.outputSize = config['output_size'] if 'output_size' in config else 6
        self.clipGradientMagn = config['clipGradientMagn'] if 'clipGradientMagn' in config else False
        self.input_mean = config['input_mean'] if 'input_mean' in config else 0.0
        self.input_std  = config['input_std'] if 'input_std' in config else 1.0
        self.init_dropout = config['init_dropout'] if 'init_dropout' in config else 0.2
        self.dropout_temp = config['dropout_temp'] if 'dropout_temp' in config else 10.0
        self.n_reps = config['n_reps'] if 'n_reps' in config else 1
        self.module_chain = module_chain
        self.name = "Torch MLP Generator Model"

        if isinstance(self.nNodes, int):
            self.nNodes = [self.nNodes for i in range(self.nLayers)]
        if isinstance(self.activations, str):
            self.activations = [self.activations for i in range(self.nLayers)]
        

        assert len(self.nNodes) == self.nLayers, f"Generator: A list/array of number of nodes for each hidden layer was passed, but it's length (={len(self.nNodes)}) does not match number of layers (={self.nLayers})"
        assert len(self.activations) == self.nLayers, f"Generator: A list of activations for each hidden layer was passed, but it's length (={len(self.activations)}) does not match number of layers (={self.nLayers})"
       
        self.layers = OrderedDict()
        prevSize = self.inputSize
        dropout_layer = MCDropout(self.init_dropout,self.dropout_temp)
        for i in range(self.nLayers):
            self.layers['layer'+str(i)] = Linear(prevSize, self.nNodes[i], bias=self.useBias[i])
            if self.activations[i].lower() == "leakyrelu":
                self.layers['Activation'+str(i)] = LeakyReLU(0.2)
            if self.activations[i].lower() == "relu":
                self.layers['Activation'+str(i)] = ReLU()
            if self.activations[i].lower() == "tanh":
                self.layers['Activation'+str(i)] = Tanh()
            if self.activations[i].lower() == "sigmoid":
                self.layers['Activation'+str(i)] = Sigmoid()

            if self.batchNorms[i]:
                self.layers['BatchNorm'+str(i)] = BatchNorm1d(self.nNodes[i])

            self.layers['Dropout'+str(i)] = dropout_layer

            prevSize = self.nNodes[i]
        self.layers['output_layer'] = torch.nn.Linear(prevSize, self.outputSize, bias=False)
        self.layers['output_activation'] = Sigmoid()

        self.model = Sequential(self.layers).to(self.devices)
        if self.applyCustomWeightsInit:
            self.model.apply(_custom_weights_init)

        self.loss_function = torch.nn.MSELoss()

    def forward(self, input):
        """
            Forward pass of the generator module to produce parameters

        Inputs
        ------
            input: tensor or numpy nd array
                input that is to be fed into the model

        Returns
        -------
            params: tensor
                Generated fake parameters
        """
        return self.model((input - self.input_mean)/self.input_std)

    def generate(self,x):
        """
            Generate the parameters for the workflow this corresponds to: predict_uq
        """

        # Technically, one could use the: torch.repeat_interleaf() function and repeat the input-vector n_rep times
        # However, we want to ensure that the MC-dropout is utilized properly so we call the model n_rep times and
        # concatenate the predictions... 
        prediction_list = []
        #+++++++++++++++++++
        for _ in range(self.n_reps):
            current_prediction = self.model(x)
            prediction_list.append(current_prediction)
        #+++++++++++++++++++
        predictions = torch.cat(prediction_list,dim=0)
        
        
        par_mean = torch.mean(predictions,0)
        par_sigma = torch.std(predictions,0)

        return par_mean + torch.normal(mean=0.0,std=1.0,size=(x.size()[0],par_mean.size()[0]),device=self.devices) * par_sigma
    
    def compute_loss(self,x,x_pred,x_ref):
        sigma2_model = torch.square(torch.std(x_pred,dim=0)) + 1e-7
        
        #ref_mean = torch.mean((x-x_ref),dim=0)
        act_loss = torch.square(x-x_pred)
        ref_loss = torch.square(x-x_ref)

       # loss = torch.square((x-x_pred)-ref_mean) / sigma2_model
        # loss = torch.abs(act_loss-ref_loss) / sigma2_model
        # return torch.mean(loss + torch.log(sigma2_model))
    
        loss = torch.abs(act_loss-ref_loss) 
        return torch.mean(loss)

    
    def train(self,x,x_ref,real_norms,optimizer,num_events=1, data_dim=2):
        """
            Performs backpropagation and training of the generator model

        Inputs
        ------
            input: tensor or numpy nd array
                input that is to be fed into the model

        """
        

        optimizer.zero_grad()
        params = self.generate(x)
        fake_events, norm1, norm2 = self.module_chain[0].forward(params, num_events)
        fake_events = torch.squeeze(torch.reshape(torch.transpose(fake_events, 1, 2), (-1, data_dim)))

        for module in self.module_chain[1:]:
            fake_events = module.forward(fake_events)
        loss = self.compute_loss(x,fake_events,x_ref)
        loss.backward()

        # Assumming norms are 1-dimensional
        # norm_loss1 = self.loss_function(norm1,real_norms[0].to(self.devices))
        # norm_loss1.backward(retain_graph=True)
        # norm_loss2 = self.loss_function(norm2,real_norms[1].to(self.devices))
        # norm_loss2.backward()

        self.clip_gradient()
        optimizer.step()
        return [loss, 0.0,0.0]

    def clip_gradient(self):
        if self.clipGradientMagn:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipGradientMagn, norm_type=2.0, error_if_nonfinite=False)


# Define a custom dropout class:
class MCDropout(torch.nn.Module):
    
    def __init__(self,p0,temp):
        super(MCDropout, self).__init__()
        self.p_train = torch.nn.Parameter(self.calc_logit(p0),requires_grad=True)
        self.temp = temp

    def calc_logit(self,p0):
        p0_tensor = torch.tensor(p0)
        return torch.log(p0_tensor / (1. - p0_tensor))
    
    # The implementation used here has been taken from:
    # 'Concrete Dropout' Gal et al.,  arXiv:1705.07832v1 [stat.ML], 2017
    # Link to paper is here: https://doi.org/10.48550/arXiv.1705.07832
    def forward(self,x):
        eps = 1e-7
        u = torch.rand_like(x)
        pt = torch.ones_like(x)*torch.sigmoid(self.p_train)

        arg = torch.log(pt+eps) - torch.log(1.-pt+eps) + torch.log(u+eps) - torch.log(1.-u+eps)
        drop_prob = torch.sigmoid(self.temp*arg)
        
        x= torch.mul(x,(1.-drop_prob))
        x /= (1.-torch.sigmoid(self.p_train))
        return x


