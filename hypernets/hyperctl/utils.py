import yaml
import json


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return yaml.load(content, Loader=yaml.CLoader)


def load_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return json.loads(content)


def copy_item(src, dest, key):
    v = src.get(key)
    if v is not None:
        dest[key] = v


def http_portal(host, port):
    return f"http://{host}:{port}"
