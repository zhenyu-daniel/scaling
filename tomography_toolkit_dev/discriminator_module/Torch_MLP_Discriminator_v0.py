import torch
from torch.nn import Linear, LeakyReLU, ReLU, Tanh, Sigmoid, Dropout, Sequential, BatchNorm1d
from collections import OrderedDict
from tomography_toolkit_dev.core.discriminator_core import Discriminator

class Torch_MLP_Discriminator(torch.nn.Module, Discriminator):
    """
    A class to define multi-layer perceptron based discriminator for the GAN workflow.

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
            val: size of the inputs to the discriminator
        - key: 'output_size'
            val_type: int
            val: size of the output layer / number of outputs

    Methods
    -------
    forward(inputs)
        Forward pass of the inputs to produce output

    train(noise)
        Performs backpropagation and training of discriminator model

    """
    def __init__(self, config, devices="cpu"):
        super(Torch_MLP_Discriminator, self).__init__()
        self.devices = devices
        self.nLayers = int(config['num_layers']) if 'num_layers' in config else 4
        self.nNodes = config['num_nodes'] if 'num_nodes' in config else self.nLayers*[128]
        self.activations = config['activation'] if 'activation' in config else self.nLayers*["LeakyRelu"]
        self.useBias = config['use_bias'] if 'use_bias' in config else self.nLayers*[True]
        self.batchNorms = config['batchNorms'] if 'batchNorms' in config else self.nLayers*[False]
        self.dropouts = config['dropout_percents'] if 'dropout_percents' in config else self.nLayers*[0.]
        #self.applyCustomWeightsInit = config['applyCustomWeightsInit'] if 'applyCustomWeightsInit' in config else False
        self.weight_initialization = config['weight_initialization'] if 'weight_initialization' in config else (1+self.nLayers)*['default']
        self.bias_initialization = config['bias_initialization'] if 'bias_initialization' in config else (1+self.nLayers)*['default']
        self.inputSize = config['input_size'] if 'input_size' in config else 2,
        self.outputSize = config['output_size'] if 'output_size' in config else 1,
        self.learning_rate = config['learning_rate'] if 'learning_rate' in config else 1e-4
        self.clipGradientMagn = config['clipGradientMagn'] if 'clipGradientMagn' in config else False
        self.name = "Torch MLP Discriminator Model"
        if isinstance(self.nNodes, int):
            self.nNodes = [int(self.nNodes) for i in range(self.nLayers)]
        if isinstance(self.activations, str):
            self.activations = [self.activations for i in range(self.nLayers)]
        if isinstance(self.dropouts, float):
            self.dropouts = [self.dropouts for i in range(self.nLayers)]

        assert len(self.nNodes) == self.nLayers, f"Discriminator: A list/array of number of nodes for each hidden layer was passed, but it's length (={len(self.nNodes)}) does not match number of layers (={self.nLayers})"
        assert len(self.activations) == self.nLayers, f"Discriminator: A list of activations for each hidden layer was passed, but it's length (={len(self.activations)}) does not match number of layers (={self.nLayers})"
        assert len(self.dropouts) == self.nLayers, f"Discriminator: A list/array of dropout percentage for each hidden layer was passed, but it's length (={len(self.dropouts)}) does not match number of layers (={self.nLayers})"

        self.layers = OrderedDict()
        prevSize = self.inputSize[0]
        for i in range(self.nLayers):
            self.layers['layer'+str(i)] = torch.nn.Linear(prevSize, self.nNodes[i], bias=self.useBias[i])
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
        self.layers['output_layer'] = torch.nn.Linear(prevSize, self.outputSize[0], bias=False)
        self.layers['output_activation'] = torch.nn.Sigmoid()
        # Apply weight and bias initialization:
        self.initialize_layer(self.layers['output_layer'],'sigmoid',self.weight_initialization[self.nLayers],self.bias_initialization[self.nLayers])

        self.model = torch.nn.Sequential(self.layers).to(self.devices)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.loss_function = torch.nn.MSELoss()

    def forward(self, input):
        """
            Forward pass of the inputs to produce output

        Inputs
        ------
            input: tensor or numpy nd array
                input that is to be fed into the discriminator model

        Returns
        -------
            params: tensor
                classification scores for inputs, targets: 0 for fake and 1 for real
        """
        return self.model(input)

    def train(self, real_events, fake_events):
        """
            Performs backpropagation and training of discriminator model

        Inputs
        ------
            real_events: numpy nd-array or tensor
                Real event data in the shape expected by discriminator model

            fake_events: numpy nd-array or tensor
                Fake events generated from fake parameters

            NOTE: The shapes of the real_events and fake_events are the same!
        """
        assert real_events.size() == fake_events.size(), "The shapes of real events (="+str(real_events.size())+") and fake events (="+str(fake_events.size())+") passed to discriminator train function does not match"
        label = torch.zeros((real_events.size()[0],),device=self.devices)
        self.optimizer.zero_grad(set_to_none=True)

        label.fill_(1.0)
        real_preds = self.model(real_events).view(-1)
        loss_real = self.loss_function(real_preds,label)
        loss_real.backward()

        label.fill_(0.0)
        fake_preds = self.model(fake_events).view(-1)
        loss_fake = self.loss_function(fake_preds,label)
        loss_fake.backward()

        self.clip_gradient()
        self.optimizer.step()
        return [loss_real, loss_fake]

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


