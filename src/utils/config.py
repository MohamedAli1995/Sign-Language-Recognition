import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # Convert the dictionary to an object using bunch library.
    config = Bunch(config_dict)
    return config, config_dict


def processing_config(json_file):
    config, _ = get_config_from_json(json_file)
    return config
