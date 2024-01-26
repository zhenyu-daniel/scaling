import torch
import numpy as np
from torch.nn import Linear, LeakyReLU, ReLU, Tanh, Sigmoid, Dropout, Sequential, BatchNorm1d
from collections import OrderedDict

class Generator(torch.nn.Module):
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
        super(Generator, self).__init__()
        self.devices = devices
        self.nLayers = int(config['num_layers']) if 'num_layers' in config else 4
        self.nNodes = config['num_nodes'] if 'num_nodes' in config else self.nLayers*[128]
        self.activations = config['activation'] if 'activation' in config else self.nLayers*["LeakyRelu"]
        self.useBias = config['use_bias'] if 'use_bias' in config else self.nLayers*[True]
        self.batchNorms = config['batchNorms'] if 'batchNorms' in config else self.nLayers*[False]
        self.dropouts = config['dropout_percents'] if 'dropout_percents' in config else self.nLayers*[0.]
       # self.applyCustomWeightsInit = config['applyCustomWeightsInit'] if 'applyCustomWeightsInit' in config else False
        self.weight_initialization = config['weight_initialization'] if 'weight_initialization' in config else (1+self.nLayers)*['default']
        self.bias_initialization = config['bias_initialization'] if 'bias_initialization' in config else (1+self.nLayers)*['default']
        self.inputSize = config['input_size'] if 'input_size' in config else 10
        self.outputSize = config['output_size'] if 'output_size' in config else 6
        self.learning_rate = config['learning_rate'] if 'learning_rate' in config else 1e-4
        self.clipGradientMagn = config['clipGradientMagn'] if 'clipGradientMagn' in config else False
        self.input_mean = config['input_mean'] if 'input_mean' in config else 0.0
        self.input_std  = config['input_std'] if 'input_std' in config else 1.0
        self.module_chain = module_chain
        self.name = "Torch MLP Generator Model"

        if isinstance(self.nNodes, int):
            self.nNodes = [self.nNodes for i in range(self.nLayers)]
        if isinstance(self.activations, str):
            self.activations = [self.activations for i in range(self.nLayers)]
        if isinstance(self.dropouts, float):
            self.dropouts = [self.dropouts for i in range(self.nLayers)]

        assert len(self.nNodes) == self.nLayers, f"Generator: A list/array of number of nodes for each hidden layer was passed, but it's length (={len(self.nNodes)}) does not match number of layers (={self.nLayers})"
        assert len(self.activations) == self.nLayers, f"Generator: A list of activations for each hidden layer was passed, but it's length (={len(self.activations)}) does not match number of layers (={self.nLayers})"
        assert len(self.dropouts) == self.nLayers, f"Generator: A list/array of dropout percentage for each hidden layer was passed, but it's length (={len(self.dropouts)}) does not match number of layers (={self.nLayers})"

        self.layers = OrderedDict()
        prevSize = self.inputSize
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

            # Apply weight and bias initialization:
            self.initialize_layer(self.layers['layer'+str(i)],self.activations[i],self.weight_initialization[i],self.bias_initialization[i])

            if self.batchNorms[i]:
                self.layers['BatchNorm'+str(i)] = BatchNorm1d(self.nNodes[i])

            if self.dropouts[i] > 0:
                self.layers['Dropout'+str(i)] = Dropout(self.dropouts[i])

            prevSize = self.nNodes[i]
        self.layers['output_layer'] = torch.nn.Linear(prevSize, self.outputSize, bias=False)
        self.layers['output_activation'] = Sigmoid()
        # Apply weight and bias initialization:
        self.initialize_layer(self.layers['output_layer'],'sigmoid',self.weight_initialization[self.nLayers],self.bias_initialization[self.nLayers])

        self.model = Sequential(self.layers).to(self.devices)
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
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

    def generate(self, noise):
        """
            Generates and returns n events where n is defined by dimension of noise input

        Inputs
        ------
            input: tensor or numpy nd array
                input that is to be fed into the model

        Returns
        -------
            params: tensor
                Generated fake parameters

        """
        assert type(noise) == np.ndarray or torch.is_tensor(noise), "noise input to generate method is not a numpy array or torch tensor, please pass either of these two."
        assert noise.shape[1] == self.inputSize, f"The second dimension (={noise.shape[1]}) of noise input to generate method is different from expected model input dimension (={self.inputSize})"
        generator_output = self.model(noise)

        return generator_output

    def train(self, noise, real_norms, num_events=1, data_dim=2):
        """
            Performs backpropagation and training of the generator model

        Inputs
        ------
            input: tensor or numpy nd array
                input that is to be fed into the model

        """
        assert type(noise) == np.ndarray or torch.is_tensor(noise), "noise input to train method is not a numpy array or torch tensor, please pass either of these two."
        assert noise.shape[1] == self.inputSize, f"The second dimension (={noise.shape[1]}) of noise input to train method is different from expected model input dimension (={self.inputSize})"

        self.optimizer.zero_grad(set_to_none=True)
        params = self.model(noise)
        fake_events, norm1, norm2 = self.module_chain[0].forward(params, num_events)
        fake_events = torch.squeeze(torch.reshape(torch.transpose(fake_events, 1, 2), (-1, data_dim)))

        for module in self.module_chain[1:]:
            fake_events = module.forward(fake_events)
        fake_preds = fake_events.view(-1)
        label = torch.ones((fake_preds.shape[0],), device=self.devices)
        loss_fake = self.loss_function(fake_preds,label)
        loss_fake.backward(retain_graph=True)

        # Assumming norms are 1-dimensional
        norm_loss1 = self.loss_function(norm1,real_norms[0].to(self.devices))
        norm_loss1.backward(retain_graph=True)
        norm_loss2 = self.loss_function(norm2,real_norms[1].to(self.devices))
        norm_loss2.backward()

        self.clip_gradient()
        self.optimizer.step()
        return [loss_fake, norm_loss1, norm_loss2]

    def clip_gradient(self):
        if self.clipGradientMagn:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipGradientMagn, norm_type=2.0, error_if_nonfinite=False)

    def initialize_layer(self,layer,layer_activation,weight_init,bias_init):
        # Get the weights and bias first:
        w = None
        b = None

        if layer.weight is not None:
           w = layer.weight.data
        if layer.bias is not None:
           b = layer.bias.data

        # Handle weight initialization:
        if weight_init.lower() != "default" and w is not None: #--> Default here means the default pytorch implementation...
           if layer_activation.lower == 'linear' or layer_activation.lower() == 'tanh' or layer_activation.lower() == 'sigmoid' or layer_activation.lower() == 'softmax':
               if weight_init.lower() == 'normal':
                   torch.nn.init.xavier_normal_(w)
               if weight_init.lower() == 'uniform':
                   torch.nn.init.xavier_uniform_(w)

           if layer_activation.lower() == 'relu' or layer_activation.lower() == 'leaky_relu':
               a_slope = 0.0
               if layer_activation.lower() == 'leaky_relu':
                   a_slope = -0.2

               if weight_init.lower() == 'normal':
                  torch.nn.init.kaiming_normal_(w,a=a_slope,nonlinearity=layer_activation.lower())
               if weight_init.lower() == 'uniform':
                  torch.nn.init.kaiming_uniform_(w,a=a_slope,nonlinearity=layer_activation.lower())
        
        # Handle bias initialization: #--> Default here means the default pytorch implementation...
        if bias_init.lower() != "default" and b is not None:
            if bias_init.lower() == "normal":
                torch.nn.init.normal_(b)
            if bias_init.lower() == "uniform":
                torch.nn.init.uniform_(b)
            if bias_init.lower() == "ones":
                torch.nn.init.ones_(b)
            if bias_init.lower() == "zeros":
                torch.nn.init.zeros_(b)

