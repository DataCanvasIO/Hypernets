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


def get_job(job_name, daemon_portal):
    url_get_job = f"{daemon_portal}/api/job/{job_name}"
    data = _fetch_url(url_get_job)
    return data


def _get_job_name_and_damon_portal():
    job_name = os.getenv(consts.KEY_ENV_JOB_NAME)
    daemon_portal = os.getenv(consts.KEY_ENV_DAEMON_PORTAL)

    assert job_name
    assert daemon_portal

    return job_name, daemon_portal


def get_job_params():
    job_name, daemon_portal = _get_job_name_and_damon_portal()
    return get_job(job_name, daemon_portal)['params']


def get_job_working_dir():
    job_working_dir = os.getenv(consts.KEY_ENV_JOB_EXECUTION_WORKING_DIR)
    return job_working_dir


def list_jobs(daemon_portal):
    # if daemon_portal is None:
    #     daemon_portal = os.getenv(consts.KEY_ENV_DAEMON_PORTAL)
    assert daemon_portal
    url_get_jobs = f"{daemon_portal}/api/job"
    data = _fetch_url(url_get_jobs)
    return data['jobs']


def kill_job(daemon_portal, job_name):
    url_kill_job = f"{daemon_portal}/api/job/{job_name}/kill"
    data = _fetch_url(url_kill_job, method='post')
    return data
