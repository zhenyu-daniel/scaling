import torch
from collections import OrderedDict

"""
Sets MLP architecture, based on a config file. This script is based on the initial implementation of the Generator / Discriminator module,
done by Kishan. 
"""

class MLP_Architecture(object):

    # INITIALIZE
    #********************************
    def __init__(self,config):
        
        # Retrieve basic information:

        self.num_layers = config['num_layers'] if 'num_layers' in config else 3
        self.num_nodes = config['num_nodes'] if 'num_nodes' in config else 100
        self.activations = config['activations'] if 'activations' in config else ["LeakyRelu"] * self.num_layers
        self.output_activation = config['output_activation'] if 'output_activation' in config else "linear"
        self.weight_initializer = config['weight_initializer'] if 'weight_initializer' in config else ["glorot_normal"] * self.num_layers
        self.bias_initializer = config['bias_initializer'] if 'bias_initializer' in config else ["zeros"] * self.num_layers
        self.dropout_percents = config['dropout_percents'] if 'dropout_percents' in config else self.num_layers * [0.0]
        self.learning_rate = config['learning_rate'] if 'learning_rate' in config else 1e-4
        self.input_size = config['input_size'] if 'input_size' in config else 1
        self.output_size = config['output_size'] if 'output_size' in config else 1
        self.output_min_scale = config['output_min_scale'] if 'output_min_scale' in config else []
        self.output_max_scale = config['output_max_scale'] if 'output_max_scale' in config else []

        # Check that everything has the correct dimension:

        assert len(self.num_nodes) == self.num_layers, f"MLP: A list/array of number of nodes for each hidden layer was passed, but it's length (={len(self.num_nodes)}) does not match number of layers (={self.num_layers})"
        assert len(self.activations) == self.num_layers, f"MLP: A list of activations for each hidden layer was passed, but it's length (={len(self.activations)}) does not match number of layers (={self.num_layers})"
        assert len(self.dropout_percents) == self.num_layers, f"MLP: A list/array of dropout percentage for each hidden layer was passed, but it's length (={len(self.dropout_percents)}) does not match number of layers (={self.num_layers})"
        assert len(self.weight_initializer) == self.num_layers, f"MLP: A list/array of weight initializer for each hidden layer was passed, but it's length (={len(self.weight_initializer)}) does not match number of layers (={self.num_layers})"
        assert len(self.bias_initializer) == self.num_layers, f"MLP: A list/array of bias initializer for each hidden layer was passed, but it's length (={len(self.bias_initializer)}) does not match number of layers (={self.num_layers})"

        assert len(self.output_min_scale) == len(self.output_max_scale), f"MLP: The number of minimum output scalers (={len(self.output_min_scale)}) does not match the number of maximum output scaler (={len(self.output_max_scale)})" 

        if len(self.output_min_scale) > 0 and len(self.output_max_scale) > 0:
            assert len(self.output_min_scale) == self.output_size, f"MLP: A list/array of output minimum scalers was passed, but it's length (={len(self.output_min_scale)}) does not match output size (={self.output_size}) "
            assert len(self.output_max_scale) == self.output_size, f"MLP: A list/array of output maximum scalers was passed, but it's length (={len(self.output_max_scale)}) does not match output size (={self.output_size})"
    #********************************

    # Set activation functions for a specific layer:
    #********************************
    def set_layer_activation_function(self,activation_fn_str):
        if activation_fn_str.lower() == "leakyrelu":
            return torch.nn.LeakyReLU(0.2)

        if activation_fn_str.lower() == "relu":
            return torch.nn.ReLU()

        if activation_fn_str.lower() == "tanh":
           return torch.nn.Tanh()

        if activation_fn_str.lower() == "sigmoid":
            return torch.nn.Sigmoid()
    #********************************

    # initialize model parameters:
    #********************************
    def init_layer_parameters(self,layer_parms,initializer_str):
        if initializer_str.lower() == "normal":
            torch.nn.init.normal_(layer_parms)
        
        if initializer_str.lower() == "uniform":
            torch.nn.init.uniform_(layer_parms)

        if initializer_str.lower() == "ones":
            torch.nn.init.ones_(layer_parms)

        if initializer_str.lower() == "zeros":
            torch.nn.init.zeros_(layer_parms)

        if initializer_str.lower() == "glorot_normal":
            torch.nn.init.xavier_normal_(layer_parms)
        
        if initializer_str.lower() == "glorot_uniform":
            torch.nn.init.xavier_uniform_(layer_parms)
        
        if initializer_str.lower() == "he_normal":
            torch.nn.init.kaiming_normal_(layer_parms)
        
        if initializer_str.lower() == "he_uniform":
            torch.nn.init.kaiming_uniform_(layer_parms)
    #********************************

    # SET THE MODEL ARCHITECTURE:
    #********************************
    def set(self):
        mlp_architecture = OrderedDict()
        prevSize = self.input_size
        n_hidden_layers = len(self.num_nodes)

        # Take care of the hidden layers:
        #++++++++++++++++++++++++
        for i in range(n_hidden_layers):
            mlp_architecture['Layer'+str(i)] = torch.nn.Linear(prevSize,self.num_nodes[i])
            # Get layer parameters:
            w = mlp_architecture['Layer'+str(i)].weight
            b = mlp_architecture['Layer'+str(i)].bias

            # Set the activation function:
            mlp_architecture['Activation'+str(i)] = self.set_layer_activation_function(self.activations[i])
            
            # Add dropout, if needed:
            if self.dropout_percents[i] > 0:
                mlp_architecture['Dropout'+str(i)] = torch.nn.Dropout(self.dropout_percents[i])

            # Weight initialization:
            self.init_layer_parameters(w,self.weight_initializer[i])
            # Bias initialization:
            self.init_layer_parameters(b,self.bias_initializer[i])
            
            prevSize = self.num_nodes[i]
        #++++++++++++++++++++++++

        # Set the output layer:
        mlp_architecture['Output_Layer'] = torch.nn.Linear(prevSize,self.output_size)

        # Check the output activation:
        if self.output_activation != "linear":
            mlp_architecture['Output_Activation'] = self.set_layer_activation_function(self.output_activation)

        # Add scaling layer if scalers are provided:
        if len(self.output_max_scale) > 0 and len(self.output_min_scale) > 0:
            mlp_architecture['Scaling_Layer'] = ScalingLayer(self.output_min_scale,self.output_max_scale)

        return mlp_architecture
    #********************************

    # Debug the mode, just to be sure everything makes sense:
    #********************************
    def debug(self,architecture,config,print_message=True,model_name='MLP'):
        pass_test = True
        
        if print_message:
           print(" ")
           print("Run consistency checks on " + model_name + " architecture...")
           print(" ")

        # Check the hidden layers first:
        #+++++++++++++++++++++++++++++
        for i in range(self.num_layers):
            hl_name = 'Layer' + str(i)
            act_name = 'Activation' + str(i)

            if hl_name in architecture:
                current_layer = architecture[hl_name]
                current_activation = architecture[act_name]

                n_nodes = current_layer.out_features
                n_nodes_set = config["num_nodes"][i]
                activation_set = self.set_layer_activation_function(config["activations"][i])

                if n_nodes != n_nodes_set:
                    if print_message:
                       print(">>> WARNING: Layer " + str(i) + ": Number of nodes set (" + str(n_nodes) + ") does not match the number of nodes in configuration (" + str(n_nodes_set) + ") <<<")
                    pass_test = False

                if type(current_activation) != type(activation_set):
                    if print_message:
                       print(">>> WARNING: Layer " + str(i) + ": Activation set does not match activation in configuration <<<")
                    pass_test = False

                # If first hidden layer, check the input size:
                if i == 0:
                    n_inputs = current_layer.in_features 
                    n_inputs_set = config['input_size']

                    if n_inputs != n_inputs_set:
                        if print_message:
                           print(">>> WARNING: Input dimension (" + str(n_inputs) + ") does not match input dimention in configuration (" + str(n_inputs_set) + ") <<<")

            else:
                if print_message:
                   print(">>> WARNING: Layer: " + hl_name + " does not exist in current architecture <<<")
                pass_test = False
        #+++++++++++++++++++++++++++++

        # Check the output layer:
        if 'Output_Layer' in architecture:
            current_layer  = architecture['Output_Layer']
            n_output_nodes = current_layer.out_features
            n_output_nodes_set = config['output_size']
            output_activation = architecture['Output_Activation']

            output_activation_set = self.set_layer_activation_function(config['output_activation'])

            if n_output_nodes != n_output_nodes_set:
                if print_message:
                   print(">>> WARNING: Output Layer : Number of nodes set (" + str(n_output_nodes) + ") does not match the number of nodes in configuration (" + str(n_output_nodes_set) + ") <<<")
                pass_test = False

            if 'output_activation' in config:
                if type(output_activation) != type(output_activation_set):
                    if print_message:
                       print(">>> WARNING: Output Layer : Activation set does not match activation in configuration <<<")
                    pass_test = False

        else:
            print(">>> WARNING: Output layer does not exist in current architecture <<<")

        # Check the scaling layer:
        if 'Scaling_Layer' in architecture:
            current_layer = architecture['Scaling_Layer']
            min_scale = current_layer.min_scale
            max_scale = current_layer.max_scale

            min_scale_set = config['output_min_scale']
            max_scale_set = config['output_max_scale']

            dmin = [a-b for a,b in zip(min_scale,min_scale_set)]
            dmax = [a-b for a,b in zip(max_scale,max_scale_set)]

            if sum(dmin) != 0.0:
                if print_message:
                   print(">>> WARNING: Minimum scale does not match minimum scale in configuration <<<")
                pass_test = False

            if sum(dmax) != 0.0:
                if print_message:
                   print(">>> WARNING: Maximum scale does not match maximum scale in configuration <<<")
                pass_test = False

        else:
            if print_message:
               print(">>> INFO: No scaling layer set <<<")

        if print_message:
           print(" ")
           print("...done!")
           print(" ")

        if print_message:
           if pass_test:
               print(">>> INFO: Congratulations! Your " + model_name + " architecure passed the test! <<<")
           else:
               print(">>> INFO: Your " + model_name + " architecture did not pass the test. Please ensure that the provided architecture and configuration match. <<<")
           print(" ")

        return pass_test
    #********************************

# Define a custom scaling layer, in case need to scale the output to a specified range:
class ScalingLayer(torch.nn.Module):
      
    # Initialize:
    #***********************************************  
    def __init__(self,min_out_scale,max_out_scale):
       super().__init__()
       self.min_scale = torch.nn.Parameter(torch.FloatTensor(min_out_scale),requires_grad=False)
       self.max_scale = torch.nn.Parameter(torch.FloatTensor(max_out_scale),requires_grad=False)
    #*********************************************** 

    # Return layer:
    #*********************************************** 
    def forward(self,input):
        return (self.max_scale - self.min_scale) * input + self.min_scale
    #*********************************************** 

