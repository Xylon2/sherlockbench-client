import os
import yaml

def load_examples():
    with open("resources/investigations.yaml", 'r') as f:
        data = yaml.safe_load(f)

    return data.get('problems', {})
