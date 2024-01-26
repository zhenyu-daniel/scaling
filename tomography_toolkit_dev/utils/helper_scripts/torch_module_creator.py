import os
import sys
from tomography_toolkit_dev.cfg.module_creator_config import module_to_core_dict, core_function_dict

class Module_Creator(object):
    """
    This class attempts to create a very basic and generic module which:
    1.) Is consistent with already existing modules
    2.) Follows (most of) the 'best' practices pursued within this framework
    3.) Is easy to implement into the GAN training workflow

    Please report any issues or problems that you may encounter when using this tool
    """

    # Initialize:
    #*****************************************
    def __init__(self,module_name,class_name,registration_name):

        self.module_to_core_dict = module_to_core_dict
        self.core_function_dict = core_function_dict

        self.module_core_name = module_name.capitalize()
        self.module_folder = module_name.capitalize()
        self.module_class = class_name.capitalize()
        self.registration_name = registration_name
    
        # These are modules that require specific instructions, as they are part of the data pipeline: params --> pipeline --> physics events
        self.modules_in_pipeline =['theory','experimental','eventselection']

        # We make sure that no new core class is created, if we are working with an already existing module:
        self.use_existing_core = False
        if module_name.lower() in self.module_to_core_dict:
            if self.module_core_name.lower() == self.module_to_core_dict[module_name.lower()].lower():
                print(" ")
                print(">>> Module Creator INFO: You are trying to create a new core for an already existing module. Going to ignore the provided core name. <<<")
                print(" ")
                self.module_core_name = self.module_to_core_dict[module_name.lower()]
                self.use_existing_core = True
    #*****************************************

    # Create a skeleton core for the module:
    #*****************************************
    def create_module_core_skeleton(self):
        core_file_path = "../../core/" + self.module_core_name.lower() + "_core.py"

        # Check, if this new / updated module is part of the data pipeline:
        # Important Note: A new model is (by default) registered as a part of the data pipeline
        self.is_part_of_pipeline = self.module_folder.lower() in self.modules_in_pipeline or self.use_existing_core == False
        
        if os.path.isfile(core_file_path):
            if self.use_existing_core == False:
               print(" ")
               print(">>> Module Creator INFO: This module core already exists! Will not create a new core. <<<")
               print(" ")
        else:

            core_file = open(core_file_path,"w")
            core_file.write("from abc import ABC, abstractmethod \n")
            core_file.write("\n")
            core_file.write("\n")

            core_file.write("class " + self.module_core_name + "(ABC):\n")
            core_file.write("   # This is a skeleton for a module core class.\n")
            core_file.write("   # Inherits from abstract base class for mandating functionality (pure virtual functions). \n")
            core_file.write("\n")
            core_file.write("\n")

            core_file.write("   def __init__(self,**kwargs): \n")
            core_file.write("      pass \n")
            core_file.write("\n")
            core_file.write("\n")

            if self.use_existing_core:
                functions = self.core_function_dict()

                #+++++++++++++++++++++++++++
                for func in functions:
                    core_file.write("    @abstractmethod \n")
                    core_file.write("    def " + func + "(self):\n")
                    core_file.write("         return NotImplemented \n")
                    core_file.write("\n")
                    core_file.write("\n")
                #+++++++++++++++++++++++++++

            if self.is_part_of_pipeline == True:

                core_file.write("   # We define the .forward() function which is used for data processing during the GAN training \n")
                core_file.write("   @abstractmethod \n")
                core_file.write("   def forward(self): \n")
                core_file.write("      return NotImplemented \n")
                core_file.write("\n")
                core_file.write("\n")

                core_file.write("   # We define the .apply() function which is used for data processing outside the GAN training \n")
                core_file.write("   @abstractmethod \n")
                core_file.write("   def apply(self): \n")
                core_file.write("      return NotImplemented \n")
                core_file.write("\n")
                core_file.write("\n")

            core_file.close()
    #*****************************************
    
    # Create the module itself:
    #*****************************************
    def create_module_skeleton(self):
        module_dir = "../../" + self.module_folder.lower() + "_module"

        if self.module_core_name.lower() == "workflow":
            module_dir = "../../" + self.module_folder.lower()

        if os.path.isdir(module_dir) == False:
            os.mkdir(module_dir)

        # Get disctionary to check core functions for existing modules:
        function_dict = self.core_function_dict
       
        module_file_path = module_dir + "/" + self.module_class.lower() + '.py'
        if os.path.isfile(module_file_path):
            print(" ")
            print(">>> Module Creator WARNING: This module already exists! Will not create a new module. Going to stop here. <<<")
            print(" ")

            return 1
        else:

            module_file = open(module_file_path,"w")

            module_file.write("import torch \n")
            module_file.write("import numpy as np \n")
            module_file.write("from tomography_toolkit_dev.core." + self.module_core_name.lower() + "_core import " + self.module_core_name + "\n")
            module_file.write("\n")
            module_file.write("\n")

            module_file.write("class " + str(self.module_class) + "(" + self.module_core_name + "): \n")
            module_file.write("   # Note: This is a very generic and basic module setup. You will need to implement the functionality that you envision \n")
            module_file.write("   # Please provide a short description of this class and try to answer the following questions: \n")
            module_file.write("   # i)   What is the purpose of this class ? \n")
            module_file.write("   # ii)  What inputs are needed and what is returned ? \n")
            module_file.write("   # iii) Are there any special cases where some of this class's functionalities might not work ? \n")
            module_file.write("\n")
            module_file.write("\n")

            module_file.write("   # Initialize: \n")
            module_file.write("   #************************************************ \n")
            module_file.write("   def __init__(self,config,devices='"'cpu'"'): \n")
            module_file.write("      self.config = config #--> Configuration for this module \n")
            module_file.write("      self.devices = devices #--> Devices, such as CPU, GPU, ... \n")
            module_file.write("\n") 
            module_file.write("      # Access all variables that you need from config: \n")
            module_file.write("      # self.my_variable = config['"'my_variable'"'] if '"'my_variable'"' in config else 0.0 #--> Please make sure to provide a default value (if possible),\n")
            module_file.write("      # in case someone forgets to set this variable in the config \n")
            
            if self.is_part_of_pipeline == True:
                    module_file.write("\n") 
                    module_file.write("      self.is_blank = config['"'turn_this_module_off'"'] if '"'turn_this_module_off'"' in config else False #--> Please include the option to turn your module off \n")
                    module_file.write("      self.disable_training_behavior = False #--> This flag is only needed if you require the option to force \n")
                    module_file.write("      # a different behavior of this class during training. You can ignore this flag, if you do not need it. \n")
            
            module_file.write("   #************************************************ \n")
            module_file.write("\n")  
            module_file.write("\n")

            module_file.write("   # Define whatever function(s) you need: \n")
            module_file.write("   #************************************************ \n")
            module_file.write("   # This is just an example function: \n")
            module_file.write("   def some_function(self,data): \n")
            module_file.write("      return data \n")
            module_file.write("\n")  
            module_file.write("\n")
            module_file.write("   # Define more functions! \n")
            module_file.write("   def some_other_function(self,data): \n")
            module_file.write("      new_data = torch.pow(data,2) \n")
            module_file.write("      return new_data.to(self.devices) #--> Please make sure that your calcualtion is registered on the device \n") 
            module_file.write("   #************************************************ \n")
            module_file.write("\n")  
            module_file.write("\n")

            if self.module_folder.lower() in function_dict:
                functions = function_dict[self.module_folder.lower()]

                #++++++++++++++++++++++++++
                for f in functions:
                    module_file.write("   # The following are core functions from the core class which have to present in this class: \n")
                    module_file.write("   #************************************************ \n")
                    module_file.write("   def " + f + "(self,data): \n")
                    module_file.write("       return data \n")
                    module_file.write("   #************************************************ \n")  
                    module_file.write("\n")  
                    module_file.write("\n")
                #++++++++++++++++++++++++++

            if self.is_part_of_pipeline == True:

               module_file.write("   # Define forward() pass which is used during the GAN training: \n")
               module_file.write("   #************************************************ \n")
               module_file.write("   def forward(self,data): \n")
               module_file.write("       if self.is_blank == True: #--> Turn the module off during training \n")
               module_file.write("          return data.to(self.devices) #--> Just return the unaltered data \n")
               module_file.write("       # Now do whatever additional calculations you need here: \n")
               module_file.write("       # new_data = self.some_function(data) \n")
               module_file.write("       # return new_data \n")
               module_file.write("       # Maybe you have to include the option to force a different behavior during training: \n")
               module_file.write("       # if self.diable_training_bahavior == True:  \n")
               module_file.write("       #      return self.some_other_function(data) \n")
               module_file.write("   #************************************************ \n")
               module_file.write("\n")  
               module_file.write("\n")
               module_file.write("   # Define apply() pass which is used outside the GAN training: \n")
               module_file.write("   #************************************************ \n")
               module_file.write("   def apply(self,data): \n")
               module_file.write("       if self.is_blank == True: #--> Turn the module off during training \n")
               module_file.write("          return data.to(self.devices) #--> Just return the unaltered data \n")
               module_file.write("       # Now do whatever additional calculations you need here: \n")
               module_file.write("       # new_data = self.some_other_function(data) \n")
               module_file.write("       # return new_data \n")
               module_file.write("   #************************************************ \n")
               module_file.write("\n")  
               module_file.write("\n")
               module_file.write("   # Important Note(s): \n")
               module_file.write("   #------------------------------------ \n")
               module_file.write("   # (i)   We define two output functions here (.forward and .apply), because \n")
               module_file.write("   #       there might be cases where we can not use certain procedures durng the GAN training (e.g. if-conditions) \n")
               module_file.write("   #       We have to ensure that .forward() guarantees backpropagation, meaning the functions implemented here have to be differentiable \n")
               module_file.write("   # (ii)  This does not hold for .apply(). Here we can do whatever we want, because this function is called outside the training loop. \n")
               module_file.write("   # (iii) .forward() is basically an analytical approximation of .apply(), in order to ensure a gradient flow. However, if your implementations \n")
               module_file.write("   #       are not likely to cause any trouble during training, then .forward() and .apply() are exactly the same. \n")
               module_file.write("   # (iv)  You may ignore all the above, if you are designing a new module that does not require .forward() or .apply() \n")
               module_file.write("   #------------------------------------ \n")

            module_file.close()

            return 0
    #*****************************************

    # Take care of the registration:
    #*****************************************
    def handle_module_registration(self):
        registration_file_path = "../../" + self.module_folder.lower() + "_module/registration.py"

        if self.module_core_name.lower() == "workflow":
           registration_file_path = "../../" + self.module_folder.lower() + "/registration.py"

        if os.path.isfile(registration_file_path):
            print(" ")
            print(">>> Module Creator INFO: Module registration already exists! Will not create a new registration. <<<")
            print(" ")
        else:
            reg_file = open(registration_file_path,"w")
            reg_file.write("import importlib \n")
            reg_file.write("import logging \n")
            reg_file.write("\n")
            reg_file.write(self.module_folder.lower() + "_log = logging.getLogger(\""+ self.module_folder + " Registry\") \n")
            reg_file.write("\n")
            reg_file.write("def load(name): \n")
            reg_file.write("   mod_name, attr_name = name.split(\":\") \n")
            reg_file.write("   print(f'Attempting to load {mod_name} with {attr_name}') \n")
            reg_file.write("   mod = importlib.import_module(mod_name) \n")
            reg_file.write("   fn = getattr(mod, attr_name) \n")
            reg_file.write("   return fn \n")
            reg_file.write("\n")
            reg_file.write("\n")

            reg_file.write("class " + self.module_folder + "Spec(object): \n")
            reg_file.write("      def __init__(self, id, entry_point=None, kwargs=None): \n")
            reg_file.write("          self.id = id \n")
            reg_file.write("          self.entry_point = entry_point \n")
            reg_file.write("          self._kwargs = {} if kwargs is None else kwargs \n")
            reg_file.write("\n")
            reg_file.write("\n")
            reg_file.write("      def make(self, **kwargs): \n")
            reg_file.write("          # Instantiates an instance of the agent with appropriate kwargs \n")
            reg_file.write("          if self.entry_point is None: \n")
            reg_file.write("              raise " + self.module_folder.lower() + "_log.error('Attempting to make deprecated agent {}. (HINT: is there a newer registered version of this agent?)'.format(self.id)) \n")
            reg_file.write("          _kwargs = self._kwargs.copy() \n")
            reg_file.write("          _kwargs.update(kwargs) \n")
            reg_file.write("\n")
            reg_file.write("          if callable(self.entry_point): \n")
            reg_file.write("              " + self.module_folder.lower() + " = self.entry_point(**_kwargs) \n")
            reg_file.write("          else: \n")
            reg_file.write("              cls = load(self.entry_point) \n")
            reg_file.write("              " + self.module_folder.lower() + " = cls(**_kwargs) \n")
            reg_file.write("\n")
            reg_file.write("          return " + self.module_folder.lower() + "\n")
            reg_file.write("\n")
            reg_file.write("\n")

            reg_file.write("class " + self.module_folder + "Registry(object): \n")
            reg_file.write("      def __init__(self): \n")
            reg_file.write("          self." + self.module_folder.lower() + "_specs = {} \n")
            reg_file.write("\n")
            reg_file.write("      def make(self, path, **kwargs): \n")
            reg_file.write("          if len(kwargs) > 0: \n")
            reg_file.write("            " + self.module_folder.lower() + "_log.info('Making new agent: %s (%s)', path, kwargs) \n")
            reg_file.write("          else: \n")
            reg_file.write("            " + self.module_folder.lower() + "_log.info('Making new agent: %s', path) \n")
            reg_file.write("          " + self.module_folder.lower() + "_spec = self.spec(path) \n")
            reg_file.write("          " + self.module_folder.lower() + " = " + self.module_folder.lower() + "_spec.make(**kwargs) \n")
            reg_file.write("          return " + self.module_folder.lower() + "\n")
            reg_file.write("\n")
            reg_file.write("      def all(self): \n")
            reg_file.write("          return self." + self.module_folder.lower() + "_specs.values() \n")
            reg_file.write("\n")
            reg_file.write("      def spec(self, path): \n")
            reg_file.write("          if ':' in path: \n")
            reg_file.write("            mod_name, _sep, id = path.partition(':') \n")
            reg_file.write("            try: \n")
            reg_file.write("              importlib.import_module(mod_name) \n")
            reg_file.write("            except ImportError: \n")
            reg_file.write("               raise " + self.module_folder.lower() + "_log.error('A module ({}) was specified for the agent but was not found, make sure the package is installed with `pip install` before calling `exa_gym_agent.make()`'.format(mod_name)) \n")
            reg_file.write("\n")
            reg_file.write("          else: \n")
            reg_file.write("            id = path \n")
            reg_file.write("\n")
            reg_file.write("          try: \n")
            reg_file.write("            return self." + self.module_folder.lower() + "_specs[id] \n")
            reg_file.write("\n")
            reg_file.write("          except KeyError: \n")
            reg_file.write("            raise " + self.module_folder.lower() + "_log.error('No registered agent with id: {}'.format(id)) \n")
            reg_file.write("\n")
            reg_file.write("      def register(self, id, **kwargs): \n")
            reg_file.write("            if id in self." + self.module_folder.lower() + "_specs: \n")
            reg_file.write("                raise " + self.module_folder.lower() + "_log.error('Cannot re-register id: {}'.format(id)) \n")
            reg_file.write("            self." + self.module_folder.lower() + "_specs[id] = " + self.module_folder + "Spec(id, **kwargs) \n")
            reg_file.write("\n")
            reg_file.write("\n")
            reg_file.write("# Global agent registry \n")
            reg_file.write(self.module_folder.lower() + "_registry = " + self.module_folder + "Registry() \n")
            reg_file.write("\n")
            reg_file.write("\n")
            reg_file.write("def register(id, **kwargs): \n")
            reg_file.write("    return " + self.module_folder.lower() + "_registry.register(id, **kwargs) \n")
            reg_file.write("\n")
            reg_file.write("\n")
            reg_file.write("def make(id, **kwargs): \n")
            reg_file.write("    return " + self.module_folder.lower() + "_registry.make(id, **kwargs) \n")
            reg_file.write("\n")
            reg_file.write("\n")
            reg_file.write("def spec(id): \n")
            reg_file.write("    return " + self.module_folder.lower() + "_registry.spec(id) \n")


            reg_file.close()
    #*****************************************

    # Add module to the __init__.py file:
    #*****************************************
    def update_init(self):
        init_file_path = "../../" + self.module_folder.lower() + "_module/__init__.py"
        
        add_fragment = "_module."
        if self.module_core_name.lower() == "workflow":
            init_file_path = "../../" + self.module_folder.lower() + "/__init__.py"
            add_fragment = "."

        init_file = None

        if os.path.isfile(init_file_path):
           print(" ")
           print(">>> Module Creator INFO: __init__.py already exists! Will update this file with the new module. <<<")
           print(" ")

           init_file = open(init_file_path,"a")
           init_file.write("\n")
           init_file.write("\n")
        else:
           init_file = open(init_file_path,"w")
           init_file.write("from tomography_toolkit_dev." + self.module_folder.lower() + add_fragment + "registration import register, make")
           init_file.write("\n")
           init_file.write("\n")

        init_file.write("register(\n")
        init_file.write("   id=\"" + self.registration_name + "\", \n")
        init_file.write("   entry_point=\"tomography_toolkit_dev." + self.module_folder.lower() + add_fragment + self.module_class.lower() + ":" + self.module_class + "\" \n")
        init_file.write(") \n")
        init_file.write("from tomography_toolkit_dev."  + self.module_folder.lower() + add_fragment + self.module_class.lower() + " import " + self.module_class + " \n")

        init_file.close()
    #*****************************************

    # Last but not least, create a unit-test:
    #*****************************************
    def create_unit_test(self):
        utest_path = "../../../utests/utest_" + self.module_class.lower() + ".py"
        
        if os.path.isfile(utest_path):
            print(" ")
            print(">>> Module Creator WARNING: This unit-test already exists! Please check your settings. <<<")
            print(" ")
        else:
            utest_file = open(utest_path,"w")
            utest_file.write("import unittest \n")
            utest_file.write("#import unittest2 as unittest \n")

            if self.module_core_name.lower() == "workflow": 
                utest_file.write("import tomography_toolkit_dev." + self.module_folder.lower() + " as " + self.module_folder.lower() + "\n")
            else:
                utest_file.write("import tomography_toolkit_dev." + self.module_folder.lower() + "_module as " + self.module_folder.lower() + "\n")

            utest_file.write("import torch \n")
            utest_file.write("\n")
            utest_file.write("\n")

            utest_file.write("class Test_" + self.module_folder + "_Module(unittest.TestCase): \n")
            utest_file.write("   # Very important note: This is just an example for a unit-test. It is your task / responsibilty to include a proper testing \n")
            utest_file.write("   # mechanism for the new module class that you implemented. The unit-test ensures (to first order) that everything is working the \n")
            utest_file.write("   # way it should work. You basically test your own logic. \n")
            utest_file.write("\n")
            utest_file.write("\n")

            utest_file.write("   def test_some_function(self): \n")
            utest_file.write("       # Load the module: \n")
            utest_file.write("       module = " + self.module_folder.lower() + ".make(\"" + self.registration_name + "\",config={})\n")
            utest_file.write("       # Create data for testing: \n")
            utest_file.write("       test_data = torch.normal(mean=0.5,std=0.2,size=(100,2)) \n")
            utest_file.write("       # Pass data through the module, using some_function() \n")
            utest_file.write("       data_after_function = module.some_function(test_data) \n")
            utest_file.write("       # This data has to be (by construction of this simple test) equal to the initial data: \n")
            utest_file.write("       diff = test_data - data_after_function \n")
            utest_file.write("       sum_diff = torch.sum(diff).numpy() \n ")
            utest_file.write("       # sum_diff has to be zero: \n")
            utest_file.write("       self.assertEqual(sum_diff,0.0) \n")
            utest_file.write("\n")
            utest_file.write("\n")

            utest_file.write("   def test_some_other_function(self): \n")
            utest_file.write("       # Load the module: \n")
            utest_file.write("       module = " + self.module_folder.lower() + ".make(\"" + self.registration_name + "\",config={'"'turn_this_module_off'"':True})\n")
            utest_file.write("       # Create data for testing: \n")
            utest_file.write("       test_data = torch.normal(mean=0.5,std=0.2,size=(100,2)) \n")
            utest_file.write("       # Pass data through the module, using forward() \n")
            utest_file.write("       data_after_other_function = module.some_other_function(test_data) \n")
            utest_file.write("       # This data has to be (by construction of this simple test) equal to the initial data squared: \n")
            utest_file.write("       diff = torch.pow(test_data,2) - data_after_other_function \n")
            utest_file.write("       sum_diff = torch.sum(diff).numpy() \n ")
            utest_file.write("       # sum_diff has to be zero: \n")
            utest_file.write("       self.assertEqual(sum_diff,0.0) \n")
            utest_file.write("\n")
            utest_file.write("\n")

            utest_file.write("if __name__ == \"__main__\": \n")
            utest_file.write("   unittest.main() \n")
            utest_file.write("\n")
            utest_file.write("\n")

            utest_file.close()
    #*****************************************




if __name__ == '__main__':
    user_inputs = sys.argv

    print(" ")
    print("******************************")
    print("*                            *")
    print("*   Quantom Module Creator   *")
    print("*                            *")
    print("******************************")
    print(" ")

    if len(user_inputs) < 4:
        print(" ")
        print("This script requires 3 input arguments:")
        print(" ")
        print("1.) name:           Name of the module (e.g. theory, experimental, or the name for a new module)")
        print("2.) class name:     Name of the specific class you want to implement (e.g. better_calculations.py)")
        print("3.) registry_name:  Registry name (e.g. my_fancy_module)")
        print(" ")
        print("After chosing a set of proper names (i.e. please avoid characters like: '-' '?' '*' '+' etc.),")
        print("simply run:")
        print("  ")
        print("python torch_module_creator.py name class_name registry_name")
        print(" ")
        print("And you are all set!")
        print(" ")
        print("-----------------------------------------------------------------")
        print(" ")
        print("For your convenience, here is a list of already existing modules:")
        print(" ")

        #+++++++++++++++++++++++++++++
        for key in module_to_core_dict:
            print(key)
        #+++++++++++++++++++++++++++++
        
        print(" ")

    else:
        
        creator = Module_Creator(user_inputs[1],user_inputs[2],user_inputs[3])
        
        creator.create_module_core_skeleton()
    
        module_already_exists = creator.create_module_skeleton()

        if module_already_exists == 0:

           creator.handle_module_registration()

           creator.update_init()

           creator.create_unit_test()

           print(" ")
           print("Congraulations! You created a shiny new module!")
           print("Bevore you dive into your analysis, please consider the following items:")
           print("1.) Make sure to describe the implementations / changes you make for a module")
           print("2.) Check out the unit-test directory and update the unit-test script for this module (name: utest_" + user_inputs[2].lower() + '.py)')
           print("3.) Report any problems / suggestions or issues")
           print("4.) If this is a new (and approved) module, make sure to update the dictionaries in the file: 'module_creator_config'")
           print("5.) Have a great day!")
           print(" ")


