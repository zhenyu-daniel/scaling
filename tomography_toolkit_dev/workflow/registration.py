import importlib
import logging

log = logging.getLogger("Workflow Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class WorkflowSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the workflow with appropriate kwargs"""
        if self.entry_point is None:
            raise log.error('Attempting to make deprecated workflow {}. \
                               (HINT: is there a newer registered version \
                               of this workflow?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            workflow = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            workflow = cls(**_kwargs)

        return workflow


class WorkflowRegistry(object):
    def __init__(self):
        self.workflow_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            log.info('Making new Workflow: %s (%s)', path, kwargs)
        else:
            log.info('Making new Workflow: %s', path)
        specs = self.spec(path)
        workflow = specs.make(**kwargs)

        return workflow

    def all(self):
        return self.workflow_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise log.error('A module ({}) was specified for the workflow but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `workflow.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.workflow_specs[id]
        except KeyError:
            raise log.error('No registered workflow with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.workflow_specs:
            raise log.error('Cannot re-register id: {}'.format(id))
        self.workflow_specs[id] = WorkflowSpec(id, **kwargs)


# Global agent registry
workflow_registry = WorkflowRegistry()


def register(id, **kwargs):
    return workflow_registry.register(id, **kwargs)


def make(id, **kwargs):
    return workflow_registry.make(id, **kwargs)


def spec(id):
    return workflow_registry.spec(id)
