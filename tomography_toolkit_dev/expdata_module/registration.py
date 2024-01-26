import importlib
import logging

exp_log = logging.getLogger("Experimental Data Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class ExpDataSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the agent with appropriate kwargs"""
        if self.entry_point is None:
            raise exp_log.error('Attempting to make deprecated agent {}. \
                               (HINT: is there a newer registered version \
                               of this agent?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            exp = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            exp = cls(**_kwargs)

        return exp


class ExpDataRegistry(object):
    def __init__(self):
        self.exp_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            exp_log.info('Making new agent: %s (%s)', path, kwargs)
        else:
            exp_log.info('Making new agent: %s', path)
        exp_spec = self.spec(path)
        exp = exp_spec.make(**kwargs)

        return exp

    def all(self):
        return self.exp_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise exp_log.error('A module ({}) was specified for the agent but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `exa_gym_agent.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.exp_specs[id]
        except KeyError:
            raise exp_log.error('No registered agent with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.exp_specs:
            raise exp_log.error('Cannot re-register id: {}'.format(id))
        self.exp_specs[id] = ExpDataSpec(id, **kwargs)


# Global agent registry
exp_registry = ExpDataRegistry()


def register(id, **kwargs):
    return exp_registry.register(id, **kwargs)


def make(id, **kwargs):
    return exp_registry.make(id, **kwargs)


def spec(id):
    return exp_registry.spec(id)
