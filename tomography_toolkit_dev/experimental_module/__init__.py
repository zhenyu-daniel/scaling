from tomography_toolkit_dev.experimental_module.registration import register, make

register(
    id="simple_det",
    entry_point="tomography_toolkit_dev.experimental_module.Torch_Simplified_Detector:Simplified_Detector"
)

from tomography_toolkit_dev.experimental_module.Torch_Simplified_Detector import Simplified_Detector
