import importlib
import logging

evtsel_log = logging.getLogger("Event Selection Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    print(f'Attempting to load {mod_name} with {attr_name}')
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class EventSelectionSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the agent with appropriate kwargs"""
        if self.entry_point is None:
            raise evtsel_log.error('Attempting to make deprecated agent {}. \
                               (HINT: is there a newer registered version \
                               of this agent?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            evtsel = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            evtsel = cls(**_kwargs)

        return evtsel


class EventSelectionRegistry(object):
    def __init__(self):
        self.evtsel_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            evtsel_log.info('Making new agent: %s (%s)', path, kwargs)
        else:
            evtsel_log.info('Making new agent: %s', path)
        evtsel_spec = self.spec(path)
        evtsel = evtsel_spec.make(**kwargs)

        return evtsel

    def all(self):
        return self.evtsel_specs.values()

    def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise evtsel_log.error('A module ({}) was specified for the agent but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `exa_gym_agent.make()`'.format(mod_name))

        else:
            id = path

        try:
            return self.evtsel_specs[id]
        except KeyError:
            raise evtsel_log.error('No registered agent with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.evtsel_specs:
            raise evtsel_log.error('Cannot re-register id: {}'.format(id))
        self.evtsel_specs[id] = EventSelectionSpec(id, **kwargs)


# Global agent registry
evtsel_registry = EventSelectionRegistry()


def register(id, **kwargs):
    return evtsel_registry.register(id, **kwargs)


def make(id, **kwargs):
    return evtsel_registry.make(id, **kwargs)


def spec(id):
    return evtsel_registry.spec(id)
