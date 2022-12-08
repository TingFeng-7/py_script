import json
import yaml
import pandas as pd
import xml

#json yaml

def read_json_instance(name):
    return json.load(name)

def save_json_instance(content_path, save):
    with open(content_path, 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)

        
def read_yaml_instance(name):
    return yaml.load()