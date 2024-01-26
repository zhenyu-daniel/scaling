# Dictionary to check if the core class and an existing module match
module_to_core_dict = {
            'theory': 'Theory',
            'generator': 'Generator',
            'discriminator': 'Discriminator',
            'experimental': 'Simulation',
            'eventselection': 'EventSelection',
            'expdata': 'ExpData',
            'workflow': 'Workflow'
}

# Dictionary that helps to implement core functions for existing module classes:
# This is not pretty, because we have to keep track of which module classes exist and which specific functions
# (other than apply() and forward()) they are using. Better solutions or ideas are very welcome!
core_function_dict = {
            'theory': ['paramsToEventsMap'],
            'generator': ['forward','generate','train'],
            'discriminator': ['forward','train'],
            'experimental': ['apply_detector_response'],
            'eventselection': ['filter'],
            'expdata': ['load_data','return_data'],
            'workflow': ['run']
}