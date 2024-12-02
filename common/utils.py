import os
import re


def find_last_model(path: str) -> str:
    assert os.path.isdir(path), f'Path {path} is not a directory'

    model_names = [model for model in os.listdir(path) if model.endswith('.pth')]
    if model_names:
        last_model = sorted(model_names, key=lambda x: int(re.match(rf'\S+_(\d+).pth', x).groups(1)[0]))[-1]
        return last_model

    return None

def class_exists(class_name):
    return class_name in globals() and isinstance(globals()[class_name], type)
