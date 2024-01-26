from tomography_toolkit_dev.workflow.registration import register, make

register(
    id="sequencial_workflow_v0",
    entry_point="tomography_toolkit_dev.workflow:seq_workflow"
)
from tomography_toolkit_dev.workflow.sequential_workflow_v0 import seq_workflow

register(
    id="horovod_workflow_v0",
    entry_point="tomography_toolkit_dev.workflow:hvd_workflow"
)
from tomography_toolkit_dev.workflow.horovod_workflow_v0 import hvd_workflow

register(
    id="sequential_workflow_v1",
    entry_point="tomography_toolkit_dev.workflow:seq_ae_workflow"
)
from tomography_toolkit_dev.workflow.sequential_workflow_v1 import seq_ae_workflow

register(
    id="horovod_workflow_v1",
    entry_point="tomography_toolkit_dev.workflow:hvd_ae_workflow"
)
from tomography_toolkit_dev.workflow.horovod_workflow_v1 import hvd_ae_workflow
