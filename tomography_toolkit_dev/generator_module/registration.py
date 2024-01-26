import importlib
import logging

gen_log = logging.getLogger("Generator Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class GeneratorSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the agent with appropriate kwargs"""
        if self.entry_point is None:
            raise gen_log.error('Attempting to make deprecated agent {}. \
                               (HINT: is there a newer registered version \
                               of this agent?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class GeneratorRegistry(object):
    def __init__(self):
        self.gen_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            gen_log.info('Making new agent: %s (%s)', path, kwargs)
        else:
            gen_log.info('Making new agent: %s', path)
        gen_spec = self.spec(path)
        gen = gen_spec.make(**kwargs)

        return gen

    def all(self):
        return self.gen_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise gen_log.error('A module ({}) was specified for the agent but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `exa_gym_agent.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.gen_specs[id]
        except KeyError:
            raise gen_log.error('No registered agent with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.gen_specs:
            raise exp_log.error('Cannot re-register id: {}'.format(id))
        self.gen_specs[id] = GeneratorSpec(id, **kwargs)


# Global agent registry
gen_registry = GeneratorRegistry()


def register(id, **kwargs):
    return gen_registry.register(id, **kwargs)


def make(id, **kwargs):
    return gen_registry.make(id, **kwargs)


def spec(id):
    return gen_registry.spec(id)

def list_registered_modules():
    return list(gen_registry.gen_specs.keys())
