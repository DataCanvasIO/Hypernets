# -*- encoding: utf-8 -*-
import argparse
import codecs
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Optional, Awaitable

import prettytable as pt
import yaml
from tornado import ioloop
from tornado.ioloop import PeriodicCallback
from tornado.log import app_log
from tornado.web import RequestHandler, Finish, HTTPError, Application

from hypernets import __version__ as current_version
from hypernets.hyperctl import Context, set_context, get_context
from hypernets.hyperctl import consts
from hypernets.hyperctl import dao
from hypernets.hyperctl import api
from hypernets.hyperctl.batch import Batch, DaemonConf
from hypernets.hyperctl.batch import ShellJob
from hypernets.hyperctl.dao import change_job_status
from hypernets.hyperctl.executor import RemoteSSHExecutorManager, NoResourceException, SSHRemoteMachine, \
    LocalExecutorManager, ShellExecutor
from hypernets.utils import logging as hyn_logging, common as common_util

logger = hyn_logging.getLogger(__name__)


class RestResult(object):

    def __init__(self, code, body):
        self.code = code
        self.body = body

    def to_dict(self):
        return {"code": self.code, "data": self.body}

    def to_json(self):
        return json.dumps(self.to_dict())


class RestCode(object):
    Success = 0
    Exception = -1


class BaseHandler(RequestHandler):

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def _handle_request_exception(self, e):
        if isinstance(e, Finish):
            # Not an error; just finish the request without logging.
            if not self._finished:
                self.finish(*e.args)
            return
        try:
            self.log_exception(*sys.exc_info())
        except Exception:
            # An error here should still get a best-effort send_error()
            # to avoid leaking the connection.
            app_log.error("Error in exception logger", exc_info=True)
        if self._finished:
            # Extra errors after the request has been finished should
            # be logged, but there is no reason to continue to try and
            # send a response.
            return
        if isinstance(e, HTTPError):
            self.send_error_content(str(e))
        else:
            self.send_error_content(str(e))

    def send_error_content(self, msg):
        # msg = "\"%s\"" % msg.replace("\"", "\\\"")
        _s = RestResult(RestCode.Exception, str(msg))
        self.set_header("Content-Type", "application/json")
        self.finish(_s.to_json())

    def response(self, result: dict = None, code=RestCode.Success):
        rest_result = RestResult(code, result)
        self.response_json(rest_result.to_dict())

    def response_json(self, response_dict):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(response_dict, indent=4))

    def get_request_as_dict(self):
        body = self.request.body
        return json.loads(body)


class IndexHandler(BaseHandler):

    def get(self, *args, **kwargs):
        return self.finish("It's working.")


class JobHandler(BaseHandler):

    def get(self, job_name, **kwargs):
        job = dao.get_job_by_name(job_name)
        if job is None:
            self.response({"msg": "resource not found"}, RestCode.Exception)
        else:
            ret_dict = job.to_dict()
            return self.response(ret_dict)


class JobOperationHandler(BaseHandler):

    OPT_KILL = 'kill'

    def post(self, job_name, operation, **kwargs):
        # job_name
        # request_body = self.get_request_as_dict()

        if operation not in [self.OPT_KILL]:
            raise ValueError(f"unknown operation {operation} ")

        # checkout job
        job: ShellJob = dao.get_job_by_name(job_name)
        if job is None:
            raise ValueError(f'job {job_name} does not exists ')

        if operation == self.OPT_KILL:  # do kill
            logger.debug(f"trying kill job {job_name}, it's status is {job.status} ")
            # check job status
            if job.status != job.STATUS_RUNNING:
                raise RuntimeError(f"job {job_name} in not in {job.STATUS_RUNNING} status but is {job.status} ")

            # find executor and close
            em: RemoteSSHExecutorManager = get_context().executor_manager
            executor = em.get_executor(job)
            logger.debug(f"find executor {executor} of job {job_name}")
            if executor is not None:
                em.kill_executor(executor)
                logger.debug(f"write failed status file for {job_name}")
                dao.change_job_status(job, job.STATUS_FAILED)
                self.response({"msg": f"{job.name} killed"})
            else:
                raise ValueError(f"no executor found for job {job.name}")


class JobListHandler(BaseHandler):

    def get(self, *args, **kwargs):
        jobs_dict = []
        for job in dao.get_jobs():
            jobs_dict.append(job.to_dict())
        self.response({"jobs": jobs_dict})


def create_batch_manage_webapp():
    application = Application([
        (r'/hyperctl/api/job/(?P<job_name>.+)/(?P<operation>.+)', JobOperationHandler),
        (r'/hyperctl/api/job/(?P<job_name>.+)', JobHandler),
        (r'/hyperctl/api/job', JobListHandler),
        (r'/hyperctl', IndexHandler)
    ])
    return application
