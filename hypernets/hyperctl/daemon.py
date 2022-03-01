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
from hypernets.hyperctl import runtime
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

    @staticmethod
    def detect_encoding(b):
        bstartswith = b.startswith
        if bstartswith((codecs.BOM_UTF32_BE, codecs.BOM_UTF32_LE)):
            return 'utf-32'
        if bstartswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)):
            return 'utf-16'
        if bstartswith(codecs.BOM_UTF8):
            return 'utf-8-sig'

        if len(b) >= 4:
            if not b[0]:
                # 00 00 -- -- - utf-32-be
                # 00 XX -- -- - utf-16-be
                return 'utf-16-be' if b[1] else 'utf-32-be'
            if not b[1]:
                # XX 00 00 00 - utf-32-le
                # XX 00 00 XX - utf-16-le
                # XX 00 XX -- - utf-16-le
                return 'utf-16-le' if b[2] or b[3] else 'utf-32-le'
        elif len(b) == 2:
            if not b[0]:
                # 00 XX - utf-16-be
                return 'utf-16-be'
            if not b[1]:
                # XX 00 - utf-16-le
                return 'utf-16-le'
        # default
        return 'utf-8'

    def response(self, result: dict = None, code=RestCode.Success):
        rest_result = RestResult(code, result)
        self.response_json(rest_result.to_dict())

    def response_json(self, response_dict):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(response_dict, indent=4))

    def get_request_as_dict(self):
        body = self.request.body
        # compatible for py35/36
        body = body.decode(self.detect_encoding(body), 'surrogatepass')
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


class Scheduler:

    def __init__(self, exit_on_finish, interval):
        self.exit_on_finish = exit_on_finish
        self._timer = PeriodicCallback(self.schedule, interval)

    def start(self):
        self._timer.start()

    @staticmethod
    def _check_executors(executor_manager):
        finished = []
        for executor in executor_manager.waiting_executors():
            executor: ShellExecutor = executor
            if executor.status() in ShellJob.FINAL_STATUS:
                finished.append(executor)

        for finished_executor in finished:
            executor_status = finished_executor.status()
            job = finished_executor.job
            logger.info(f"job {job.name} finished with status {executor_status}")
            change_job_status(job, finished_executor.status())
            executor_manager.release_executor(finished_executor)

    @staticmethod
    def _dispatch_jobs(executor_manager, jobs):
        for job in jobs:
            if job.status != job.STATUS_INIT:
                # logger.debug(f"job '{job.name}' status is {job.status}, skip run")
                continue
            try:
                logger.debug(f'trying to alloc resource for job {job.name}')
                executor = executor_manager.alloc_executor(job)
                process_msg = f"{len(executor_manager.allocated_executors())}/{len(jobs)}"
                logger.info(f'allocated resource for job {job.name}({process_msg}), data dir at {job.job_data_dir} ')
                # os.makedirs(job.job_data_dir, exist_ok=True)
                change_job_status(job, job.STATUS_RUNNING)
                executor.run()
            except NoResourceException:
                logger.debug(f"no enough resource for job {job.name} , wait for resource to continue ...")
                break
            except Exception as e:
                change_job_status(job, job.STATUS_FAILED)
                logger.exception(f"failed to run job '{job.name}' ", e)
                continue
            finally:
                pass

    def schedule(self):
        c = get_context()
        executor_manager = c.executor_manager
        jobs = c.batch.jobs

        # check all jobs finished
        job_finished = c.batch.is_finished()
        if job_finished:
            batch_summary = json.dumps(c.batch.summary())
            logger.info("all jobs finished, stop scheduler:\n" + batch_summary)
            self._timer.stop()  # stop the timer
            if self.exit_on_finish:
                logger.info("stop ioloop")
                ioloop.IOLoop.instance().stop()
            return

        self._check_executors(executor_manager)
        self._dispatch_jobs(executor_manager, jobs)


def create_batch_manage_webapp():
    application = Application([
        (r'/api/job/(?P<job_name>.+)/(?P<operation>.+)', JobOperationHandler),
        (r'/api/job/(?P<job_name>.+)', JobHandler),
        (r'/api/job', JobListHandler),
        (r'/', IndexHandler),
        (r'', IndexHandler),
    ])
    return application


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return yaml.load(content, Loader=yaml.CLoader)


def load_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return json.loads(content)


def _copy_item(src, dest, key):
    v = src.get(key)
    if v is not None:
        dest[key] = v


def run_generate_job_specs(template, output):
    yaml_file = template
    output_path = Path(output)
    # 1. validation
    # 1.1. checkout output
    if output_path.exists():
        raise FileExistsError(output)

    # load file
    config_dict = load_yaml(yaml_file)

    # 1.3. check values should be array
    assert "params" in config_dict
    params = config_dict['params']
    for k, v in params.items():
        if not isinstance(v, list):
            raise ValueError(f"Value of param '{k}' should be list")

    # 1.4. check command exists
    assert "execution" in config_dict
    assert 'command' in config_dict['execution']

    # 2. combine params to generate jobs
    job_param_names = params.keys()
    param_values = [params[_] for _ in job_param_names]

    def make_job_dict(job_param_values):
        job_params_dict = dict(zip(job_param_names, job_param_values))
        job_dict = {
            "name": common_util.generate_short_id(),
            "params": job_params_dict,
            "execution": config_dict['execution']
        }
        _copy_item(config_dict, job_dict, 'resource')
        return job_dict

    jobs = [make_job_dict(_) for _ in itertools.product(*param_values)]

    # 3. merge to bath spec
    batch_spec = {
        "jobs": jobs,
        'name': config_dict.get('name', common_util.generate_short_id()),
        "version": config_dict.get('version', current_version)
    }

    _copy_item(config_dict, batch_spec, 'backend')
    _copy_item(config_dict, batch_spec, 'daemon')

    # 4. write to file
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', newline='\n') as f:
        f.write(json.dumps(batch_spec, indent=4))
    return batch_spec


def run_batch_from_config_file(config, batches_data_dir=None):
    config_dict = load_json(config)
    run_batch(config_dict, batches_data_dir)


def load_batch(batch_spec_dict, batches_data_dir):
    batch_name = batch_spec_dict['name']
    jobs_dict = batch_spec_dict['jobs']

    default_daemon_conf = consts.default_daemon_conf()
    user_daemon_conf = batch_spec_dict.get('daemon')
    if user_daemon_conf is not None:
        default_daemon_conf.update(user_daemon_conf)

    batch = Batch(batch_name, batches_data_dir, DaemonConf(**default_daemon_conf))
    for job_dict in jobs_dict:
        batch.add_job(**job_dict)
    return batch


def load_batch_data_dir(batches_data_dir, batch_name):
    spec_file_path = Path(batches_data_dir) / batch_name / Batch.FILE_SPEC
    if not spec_file_path.exists():
        raise RuntimeError(f"batch {batch_name} not exists ")
    batch_spec_dict = load_json(spec_file_path)
    return load_batch(batch_spec_dict, batches_data_dir)


def run_batch(config_dict, batches_data_dir=None):
    # add batch name
    if config_dict.get('name') is None:
        batch_name = common_util.generate_short_id()
        logger.debug(f"generated batch name {batch_name}")
        config_dict['name'] = batch_name

    # add job name
    jobs_dict = config_dict['jobs']
    for job_dict in jobs_dict:
        if job_dict.get('name') is None:
            job_name = common_util.generate_short_id()
            logger.debug(f"generated job name {job_name}")
            job_dict['name'] = job_name

    batches_data_dir = get_batches_data_dir(batches_data_dir)
    batches_data_dir = Path(batches_data_dir)

    batch = load_batch(config_dict, batches_data_dir)

    logger.info(f"batches_data_path: {batches_data_dir.absolute()}")
    logger.info(f"batch name: {batch.name}")

    # check jobs status
    for job in batch.jobs:
        if job.status != job.STATUS_INIT:
            if job.status == job.STATUS_RUNNING:
                logger.warning(f"job '{job.name}' status is {job.status} in the begining,"
                               f"it may have run and will not run again this time, "
                               f"you can remove it's status file and working dir to retry the job")
            else:
                logger.info(f"job '{job.name}' status is {job.status} means it's finished, skip to run ")
            continue

    # prepare batch data dir
    if batch.data_dir_path().exists():
        logger.info(f"batch {batch.name} already exists, run again")
    else:
        os.makedirs(batch.data_dir_path(), exist_ok=True)

    # write batch config
    batch_spec_file_path = batch.spec_file_path()
    with open(batch_spec_file_path, 'w', newline='\n') as f:
        json.dump(config_dict, f, indent=4)

    # create executor manager
    backend_config = config_dict.get('backend')
    if backend_config is None:  # set default backend
        backend_config = {
            'type': 'local',
            'conf': {}
        }

    backend_type = backend_config['type']
    if backend_type == 'remote':
        remote_backend_config = backend_config['conf']
        machines = [SSHRemoteMachine(_) for _ in remote_backend_config['machines']]
        executor_manager = RemoteSSHExecutorManager(machines)
    elif backend_type == 'local':
        executor_manager = LocalExecutorManager()
    else:
        raise ValueError(f"unknown backend {backend_type}")

    # set context
    c = Context(executor_manager, batch)
    set_context(c)

    # write pid file
    with open(batch.pid_file_path(), 'w', newline='\n') as f:
        f.write(str(os.getpid()))

    # create web app
    logger.info(f"start daemon server at: {batch.daemon_conf.portal}")
    create_batch_manage_webapp().listen(batch.daemon_conf.port)

    # start scheduler
    Scheduler(batch.daemon_conf.exit_on_finish, 5000).start()

    # run io loop
    ioloop.IOLoop.instance().start()


def get_batches_data_dir(batches_data_dir):
    if batches_data_dir is None:
        bdd_env = os.environ.get(consts.KEY_ENV_BATCHES_DATA_DIR)
        if bdd_env is None:
            bdd_default = Path("~/hyperctl-batches-data-dir").expanduser().as_posix()
            logger.debug(f"use default batches_data_dir path: {bdd_default}")
            return bdd_default
        else:
            logger.debug(f"found batches_data_dir setting in environment: {bdd_env}")
            return bdd_env
    else:
        return batches_data_dir


def run_show_batches(batches_data_dir=None):
    """
    Parameters
    ----------
    batches_data_dir
        if not spec, get from environment , if is None default value is '~/hyperctl-batches'

    Returns
    -------

    """
    batches_data_dir = Path(get_batches_data_dir(batches_data_dir))

    if not batches_data_dir.exists():
        print("null")
        return

    # batches_data_dir_path = Path(batches_data_dir)
    # batches_name =
    batches_name = []
    batches_summary = []
    for filename in os.listdir(batches_data_dir):
        if (Path(batches_data_dir)/filename).is_dir():
            batches_name.append(filename)
            batch = load_batch_data_dir(batches_data_dir, filename)

            batch_summary = batch.summary()
            batches_summary.append(batch_summary)

    if len(batches_summary) == 0:
        print("empty")
        return

    # to csv format
    headers = batches_summary[0].keys()
    tb = pt.PrettyTable(headers)
    for batch_summary_ in batches_summary:
        row = [batch_summary_.get(header) for header in headers]
        tb.add_row(row)
    print(tb)


def run_show_jobs(batch_name, batches_data_dir=None):
    batches_data_dir = Path(get_batches_data_dir(batches_data_dir))
    batch = load_batch_data_dir(batches_data_dir, batch_name)
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    daemon_portal = batch.daemon_conf.portal

    jobs_dict = runtime.list_jobs(daemon_portal)

    headers = ['name', 'status']
    tb = pt.PrettyTable(headers)
    for job_dict in jobs_dict:
        row = [job_dict.get(header) for header in headers]
        tb.add_row(row)
    print(tb)


def run_kill_job(batch_name, job_name, batches_data_dir=None):
    batches_data_dir = Path(get_batches_data_dir(batches_data_dir))
    batch = load_batch_data_dir(batches_data_dir, batch_name)
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    daemon_portal = batch.daemon_conf.portal

    jobs_dict = runtime.kill_job(daemon_portal, job_name)
    print("Killed")


def show_job(batch_name, job_name, batches_data_dir=None):
    batches_data_dir = Path(get_batches_data_dir(batches_data_dir))
    batch = load_batch_data_dir(batches_data_dir, batch_name)
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    daemon_portal = batch.daemon_conf.portal

    job_dict = runtime.get_job(job_name, daemon_portal)
    job_desc = json.dumps(job_dict, indent=4)
    print(job_desc)


def main():

    bdd_help = f"batches data dir, default get from environment variable {consts.KEY_ENV_BATCHES_DATA_DIR}"

    def setup_global_args(global_parser):
        # console output
        logging_group = global_parser.add_argument_group('Console outputs')

        logging_group.add_argument('--log-level', type=str, default='INFO',
                                   help='logging level, default is %(default)s')
        logging_group.add_argument('-error', dest='log_level', action='store_const', const='ERROR',
                                   help='alias of "--log-level=ERROR"')
        logging_group.add_argument('-warn', dest='log_level', action='store_const', const='WARN',
                                   help='alias of "--log-level=WARN"')
        logging_group.add_argument('-info', dest='log_level', action='store_const', const='INFO',
                                   help='alias of "--log-level=INFO"')
        logging_group.add_argument('-debug', dest='log_level', action='store_const', const='DEBUG',
                                   help='alias of "--log-level=DEBUG"')

    def setup_batch_parser(operation_parser):
        exec_parser = operation_parser.add_parser("batch", help="batch operations")
        batch_subparsers = exec_parser.add_subparsers(dest="batch_operation")
        batch_list_parse = batch_subparsers.add_parser("list", help="list batches")
        batch_list_parse.add_argument("--batches-data-dir", help=bdd_help, default=None, required=False)

    def setup_job_parser(operation_parser):

        exec_parser = operation_parser.add_parser("job", help="job operations")

        batch_subparsers = exec_parser.add_subparsers(dest="job_operation")

        job_list_parse = batch_subparsers.add_parser("list", help="list jobs")
        job_list_parse.add_argument("-b", "--batch-name", help="batch name", default=None, required=True)
        job_list_parse.add_argument("--batches-data-dir", help=bdd_help, default=None, required=False)

        def add_job_spec_args(parser_):
            parser_.add_argument("-b", "--batch-name", help="batch name", default=None, required=True)
            parser_.add_argument("-j", "--job-name", help="job name", default=None, required=True)
            parser_.add_argument("--batches-data-dir", help=bdd_help, default=None, required=False)

        job_kill_parse = batch_subparsers.add_parser("kill", help="kill job")
        add_job_spec_args(job_kill_parse)

        job_describe_parse = batch_subparsers.add_parser("describe", help="describe job")
        add_job_spec_args(job_describe_parse)

    def setup_run_parser(operation_parser):
        exec_parser = operation_parser.add_parser("run", help="run jobs")
        exec_parser.add_argument("-c", "--config", help="specific jobs json file", default=None, required=True)
        exec_parser.add_argument("--batches-data-dir", help=bdd_help, default=None, required=False)

    def setup_generate_parser(operation_parser):
        exec_parser = operation_parser.add_parser("generate", help="generate specific jobs json file ")
        exec_parser.add_argument("-t", "--template", help="template yaml file", default=None, required=True)
        exec_parser.add_argument("-o", "--output", help="output json file", default="batch.json", required=False)

    parser = argparse.ArgumentParser(prog="hyperctl",
                                     description='hyperctl command is used to manage jobs', add_help=True)
    setup_global_args(parser)

    subparsers = parser.add_subparsers(dest="operation")

    setup_run_parser(subparsers)
    setup_generate_parser(subparsers)
    setup_batch_parser(subparsers)
    setup_job_parser(subparsers)

    args_namespace = parser.parse_args()

    kwargs = args_namespace.__dict__.copy()

    log_level = kwargs.pop('log_level')
    if log_level is None:
        log_level = hyn_logging.INFO
    hyn_logging.set_level(log_level)

    operation = kwargs.pop('operation')

    if operation == 'run':
        run_batch_from_config_file(**kwargs)
    elif operation == 'generate':
        run_generate_job_specs(**kwargs)
    elif operation == 'batch':
        batch_operation = kwargs.pop('batch_operation')
        if batch_operation == 'list':
            run_show_batches(**kwargs)
        else:
            raise ValueError(f"unknown batch operation: {batch_operation} ")
    elif operation == 'job':
        job_operation = kwargs.pop('job_operation')
        if job_operation == 'list':
            run_show_jobs(**kwargs)
        elif job_operation == 'kill':
            run_kill_job(**kwargs)
        elif job_operation == 'describe':
            show_job(**kwargs)
        else:
            raise ValueError(f"unknown job operation: {job_operation} ")
    else:
        parser.print_help()
        # raise ValueError(f"unknown job operation: {operation} ")


if __name__ == '__main__':
    main()
