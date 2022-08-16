from pathlib import Path
from typing import Optional

import yaml
import json
import requests


def load_yaml(file_path):

    if not Path(file_path).exists():
        raise FileNotFoundError(file_path)

    with open(file_path, 'r') as f:
        content = f.read()
    return yaml.load(content, Loader=yaml.CLoader)


def load_json(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(file_path)

    with open(file_path, 'r') as f:
        content = f.read()
    return json.loads(content)


def copy_item(src, dest, key):
    v = src.get(key)
    if v is not None:
        dest[key] = v


def http_portal(host, port):
    return f"http://{host}:{port}"


def get_request(url):
    def f(url_, request_data_: str):
        return requests.get(url_)

    return _request(url, f, None)


def post_request(url, request_data: Optional[str]):
    def f(url_, request_data_: str):
        return requests.post(url_, data=request_data_)

    return _request(url, f, request_data)


def _request(url,  req_func, request_data=None):
    from hypernets.utils import logging as hyn_logging
    logger = hyn_logging.getLogger(__name__)

    logger.debug(f"request data :\n{request_data}\nto {url}")
    resp = req_func(url, request_data)
    txt_resp = resp.text
    logger.debug(f"response text: \n{txt_resp}")
    json_resp = json.loads(txt_resp)
    code = json_resp['code']
    if code == 0:
        return json_resp['data']
    else:
        raise RuntimeError(txt_resp)
