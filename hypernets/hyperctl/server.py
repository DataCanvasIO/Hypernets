# -*- encoding: utf-8 -*-
import json
import sys
from typing import Optional, Awaitable

from tornado.log import app_log
from tornado.web import RequestHandler, Finish, HTTPError, Application

from hypernets.hyperctl.batch import Batch
from hypernets.hyperctl.batch import _ShellJob
from hypernets.hyperctl.executor import RemoteSSHExecutorManager
from hypernets.hyperctl.scheduler import JobScheduler
from hypernets.hyperctl.utils import http_portal
from hypernets.utils import logging as hyn_logging
import copy

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
        return self.finish("Welcome to hyperctl.")


def to_job_detail(job, batch):
    # add 'status' to return dict
    job_dict = job.to_config()
    config_dict = copy.copy(job_dict)
    config_dict['status'] = job.status
    return config_dict


class JobHandler(BaseHandler):

    def get(self, job_name, **kwargs):
        job = self.batch.get_job_by_name(job_name)
        if job is None:
            self.response({"msg": "resource not found"}, RestCode.Exception)
        else:
            ret_dict = to_job_detail(job, self.batch)
            return self.response(ret_dict)

    def initialize(self, batch: Batch):
        self.batch = batch


class JobOperationHandler(BaseHandler):

    OPT_KILL = 'kill'

    def post(self, job_name, operation, **kwargs):
        # kill job
        if operation == self.OPT_KILL:
            self.job_scheduler.kill_job(job_name)
            self.response({"msg": f"{job_name} killed"})
        else:
            raise ValueError(f"unknown operation {operation} ")

    def initialize(self, batch: Batch, job_scheduler: JobScheduler):
        self.batch = batch
        self.job_scheduler = job_scheduler


class JobListHandler(BaseHandler):

    def get(self, *args, **kwargs):
        jobs_dict = []
        for job in self.batch.jobs:
            jobs_dict.append(to_job_detail(job, self.batch))
        self.response({"jobs": jobs_dict})

    def initialize(self, batch: Batch):
        self.batch = batch


class HyperctlWebApplication(Application):

    def __init__(self, host="localhost", port=8060, **kwargs):
        self.host = host
        self.port = port
        super().__init__(**kwargs)

    @property
    def portal(self):
        return http_portal(self.host, self.port)


def create_batch_manage_webapp(server_host, server_port, batch, job_scheduler) -> HyperctlWebApplication:
    handlers = create_hyperctl_handlers(batch, job_scheduler)
    application = HyperctlWebApplication(host=server_host, port=server_port, handlers=handlers)
    return application


def create_hyperctl_handlers(batch, job_scheduler):
    handlers = [
        (r'/hyperctl/api/job/(?P<job_name>.+)/(?P<operation>.+)',
         JobOperationHandler, dict(batch=batch, job_scheduler=job_scheduler)),
        (r'/hyperctl/api/job/(?P<job_name>.+)', JobHandler, dict(batch=batch)),
        (r'/hyperctl/api/job', JobListHandler, dict(batch=batch)),
        (r'/hyperctl', IndexHandler)
    ]
    return handlers
