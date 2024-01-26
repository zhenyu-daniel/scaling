from tomography_toolkit_dev.discriminator_module.registration import register, make, list_registered_modules

register(
    id="torch_mlp_discriminator_v0",
    entry_point="tomography_toolkit_dev.discriminator_module.Torch_MLP_Discriminator_v0:Torch_MLP_Discriminator"
)
from tomography_toolkit_dev.discriminator_module.Torch_MLP_Discriminator_v0 import Torch_MLP_Discriminator

register(
    id="torch_mlp_discriminator_hvd",
    entry_point="tomography_toolkit_dev.discriminator_module.Torch_MLP_Discriminator_hvd:Torch_MLP_Discriminator"
)
from tomography_toolkit_dev.discriminator_module.Torch_MLP_Discriminator_hvd import Torch_MLP_Discriminator
