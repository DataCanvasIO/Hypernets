import json
import os

import requests

from hypernets.hyperctl import consts
from hypernets.utils import logging as hyn_logging

logger = hyn_logging.get_logger(__name__)


def _fetch_url(url, method='get'):
    logger.debug(f"http {method} to {url}")
    http_method = getattr(requests, method)
    resp = http_method(url)
    txt_resp = resp.text
    logger.debug(f"response text: \n{txt_resp}")
    json_resp = json.loads(txt_resp)
    code = json_resp['code']
    if code == 0:
        return json_resp['data']
    else:
        raise RuntimeError(txt_resp)


def get_job(job_name, api_server_portal):
    url_get_job = f"{api_server_portal}/hyperctl/api/job/{job_name}"
    data = _fetch_url(url_get_job)
    return data


def _get_job_name_and_damon_portal():
    job_name = os.getenv(consts.KEY_ENV_JOB_NAME)
    api_server_portal = f"{os.getenv(consts.KEY_ENV_SERVER_PORTAL)}"

    assert job_name
    assert api_server_portal

    return job_name, api_server_portal


def get_job_params():
    job_name, api_server_portal = _get_job_name_and_damon_portal()
    return get_job(job_name, api_server_portal)['params']


def get_job_working_dir():
    job_working_dir = os.getenv(consts.KEY_ENV_JOB_WORKING_DIR)
    return job_working_dir


def list_jobs(api_server_portal):
    # if api_server_portal is None:
    #     api_server_portal = os.getenv(consts.KEY_ENV_api_server_portal)
    assert api_server_portal
    url_get_jobs = f"{api_server_portal}/hyperctl/api/job"
    data = _fetch_url(url_get_jobs)
    return data['jobs']


def kill_job(api_server_portal, job_name):
    url_kill_job = f"{api_server_portal}/hyperctl/api/job/{job_name}/kill"
    data = _fetch_url(url_kill_job, method='post')
    return data

