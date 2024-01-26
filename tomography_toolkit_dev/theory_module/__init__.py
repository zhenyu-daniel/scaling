from tomography_toolkit_dev.theory_module.registration import register, make, list_registered_modules

register(
    id="torch_proxy_theory_v0",
    entry_point="tomography_toolkit_dev.theory_module:proxy_theory"
)
from tomography_toolkit_dev.theory_module.torch_proxy_theory_v0 import proxy_theory
