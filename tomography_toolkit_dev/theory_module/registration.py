import importlib
import logging

theory_log = logging.getLogger("Theory Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class TheorySpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the theory with appropriate kwargs"""
        if self.entry_point is None:
            raise theory_log.error('Attempting to make deprecated theory {}. \
                               (HINT: is there a newer registered version \
                               of this agent?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            theory = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            theory = cls(**_kwargs)

        return theory


class TheoryRegistry(object):
    def __init__(self):
        self.theory_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            theory_log.info('Making new theory: %s (%s)', path, kwargs)
        else:
            theory_log.info('Making new theory: %s', path)
        specs = self.spec(path)
        theory = specs.make(**kwargs)

        return theory

    def all(self):
        return self.theory_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise theory_log.error('A module ({}) was specified for the theory but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `exa_gym_agent.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.theory_specs[id]
        except KeyError:
            raise theory_log.error('No registered theory with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.theory_specs:
            raise theory_log.error('Cannot re-register id: {}'.format(id))
        self.theory_specs[id] = TheorySpec(id, **kwargs)


# Global theory registry
theory_registry = TheoryRegistry()


def register(id, **kwargs):
    return theory_registry.register(id, **kwargs)


def make(id, **kwargs):
    return theory_registry.make(id, **kwargs)


def spec(id):
    return theory_registry.spec(id)

def list_registered_modules():
    return list(theory_registry.theory_specs.keys())
