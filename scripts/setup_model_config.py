import json
import sys
import os

args = sys.argv
model_path = args[1]
config_path = os.path.join(model_path, "config.json")
abs_path = os.path.join(args[2], "elmoformanylangs", "configs")

config_dict = json.load(open(config_path, "r"))
config_type = os.path.split(config_dict['config_path'])[-1]
config_dict['config_path'] = os.path.join(abs_path, config_type)
json.dump(config_dict, open(config_path, "w"))