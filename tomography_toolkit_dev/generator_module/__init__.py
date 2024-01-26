from tomography_toolkit_dev.generator_module.registration import register, make, list_registered_modules

register(
    id="torch_mlp_generator_v0",
    entry_point="tomography_toolkit_dev.generator_module.Torch_MLP_Generator_v0:Torch_MLP_Generator"
)
from tomography_toolkit_dev.generator_module.Torch_MLP_Generator_v0 import Torch_MLP_Generator

register(
    id="torch_mlp_generator_hvd",
    entry_point="tomography_toolkit_dev.generator_module.Torch_MLP_Generator_hvd:Torch_MLP_Generator"
)
from tomography_toolkit_dev.generator_module.Torch_MLP_Generator_hvd import Torch_MLP_Generator

register(
    id="torch_mlp_encoder_hvd",
    entry_point="tomography_toolkit_dev.generator_module.Torch_MLP_Encoder_hvd:Torch_MLP_Encoder"
)
from tomography_toolkit_dev.generator_module.Torch_MLP_Encoder_hvd import Torch_MLP_Encoder
