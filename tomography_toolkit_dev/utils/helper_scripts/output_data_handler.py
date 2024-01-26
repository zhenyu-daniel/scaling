import os
import json
import subprocess

class Output_Data_Handler(object):
    """
    A collection of functions that help to handle the output data from the GAN training cycle:

    1.) Create folders where the results from the GAN training or the models themselves are stored. 

    2.) Translate configurations to either .json or .py files. This helps to keep track of the settings that have been used to run the entire GAN cycle.

    3.) Write out git hashes
    """

    # INITIALIZE

    #********************************
    def __init__(self):
        self.data = []
    #********************************

    # CREATE A FOLDER WHERE RESULTS / MODELS / CONFIGURATION FILES ARE STORED:

    #********************************
    def create_output_data_folder(self,folder_name,top_level_dir=None):
        full_path = folder_name

        if top_level_dir is not None and top_level_dir != "" and top_level_dir != " ":
            full_path = top_level_dir + '/' + folder_name

        if os.path.exists(full_path) == False:
            os.mkdir(full_path)

        return full_path
    #********************************


    # WRITE / READ CONFIGURATIONS TO / FROM A JSON FILE:
    
    #********************************
    # Write:
    def write_config_to_json_file(self,full_path_file_name,config):
        with open(full_path_file_name + ".json", "w") as jsonfile:
            json.dump(config, jsonfile)

    #-------------------

    # Read:
    def read_config_from_json_file(self,full_path_file_name):
        with open(full_path_file_name + ".json", "r") as jsonfile:
            config = json.load(jsonfile)

            return config
    #********************************

    # WRITE CONFIGURATIONS TO .py FILE: (We then read the file via 'import')

    #********************************
    def write_config_to_py_file(self,full_path_file_name,dict_list):
        with open(full_path_file_name + ".py", "w") as cfg_file:
            
            #+++++++++++++++++++++++++++++++++
            for dict in dict_list:
                conf = dict[1]
                cfg_file.write(dict[0] + "=")
                cfg_file.write(repr(conf))
                cfg_file.write("\n")
                cfg_file.write("\n")
            #+++++++++++++++++++++++++++++++++

            cfg_file.close()
    #********************************

    # STORE GIT HASHES:

    #********************************
    # Full:
    def get_git_revision_hash(self) -> str:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    
    #------------------

    # Short:
    def get_git_revision_short_hash(self) -> str:
        return subprocess.check_output(['git', 'rev-parse','--short', 'HEAD']).decode('ascii').strip()
    #********************************