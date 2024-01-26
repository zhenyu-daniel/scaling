from tomography_toolkit_dev.expdata_module.registration import register, make

register(
    id="numpy_data_parser",
    entry_point="tomography_toolkit_dev.expdata_module.Torch_Data_from_Numpy:Numpy_Data_Parser"
)

from tomography_toolkit_dev.expdata_module.Torch_Data_from_Numpy import Numpy_Data_Parser
