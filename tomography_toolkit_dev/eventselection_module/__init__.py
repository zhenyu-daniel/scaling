from tomography_toolkit_dev.eventselection_module.registration import register, make

register(
    id="rectangular_cuts",
    entry_point="tomography_toolkit_dev.eventselection_module.Torch_Rectangular_Cuts:Rectangular_Cuts"
)

from tomography_toolkit_dev.eventselection_module.Torch_Rectangular_Cuts import Rectangular_Cuts
