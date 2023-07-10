# -*- encoding: utf-8 -*-
import argparse
import itertools
import json
import os
from pathlib import Path

import prettytable as pt

from hypernets import __version__ as current_version
from hypernets.hyperctl import api
from hypernets.hyperctl import consts
from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import Batch
from hypernets.hyperctl.utils import load_yaml, load_json, copy_item
from hypernets.utils import logging
from hypernets.utils import logging as hyn_logging, common as common_util

logger = logging.getLogger(__name__)


def get_default_batches_data_dir():
    bdd_env = os.environ.get(consts.KEY_ENV_BATCHES_DATA_DIR)
    if bdd_env is None:
        bdd_default = Path("~/hyperctl-batches-data-dir").expanduser().as_posix()
        return bdd_default
    else:
        return bdd_env


def run_batch_config(config_dict, batches_data_dir):
    # add batch name
    batch_name = config_dict.get('name')
    if batch_name is None:
        batch_name = common_util.generate_short_id()
        logger.debug(f"generated batch name {batch_name}")

    # add job name
    jobs_dict = config_dict['jobs']
    for job_dict in jobs_dict:
        if job_dict.get('name') is None:
            job_name = common_util.generate_short_id()
            logger.debug(f"generated job name {job_name}")
            job_dict['name'] = job_name

    app = BatchApplication.load(config_dict, batches_data_dir)

    app.start()


def run_generate_job_specs(template, output):
    yaml_file = template
    output_path = Path(output)
    # 1. validation
    # 1.1. checkout output
    if output_path.exists():
        raise FileExistsError(output)

    # load file
    config_dict = load_yaml(yaml_file)

    # 1.3. check values should be a list
    assert "params" in config_dict

    params = config_dict['params']
    for k, v in params.items():
        if not isinstance(v, list):
            raise ValueError(f"Value of param '{k}' should be list")

    # 1.4. check command exists
    assert "command" in config_dict

    # assert "data_dir" in config_dict
    # assert "working_dir" in config_dict

    # 2. combine params to generate jobs
    job_param_names = params.keys()
    param_values = [params[_] for _ in job_param_names]

    def make_job_dict(job_param_values):
        job_params_dict = dict(zip(job_param_names, job_param_values))
        job_dict = {
            "name": common_util.generate_short_id(),
            "params": job_params_dict
        }
        copy_item(config_dict, job_dict, 'resource')
        copy_item(config_dict, job_dict, 'data_dir')
        copy_item(config_dict, job_dict, 'working_dir')
        return job_dict

    jobs = [make_job_dict(_) for _ in itertools.product(*param_values)]

    # 3. merge to bath spec
    batch_spec = {
        "jobs": jobs,
        'name': config_dict.get('name', common_util.generate_short_id()),
        "version": config_dict.get('version', current_version)
    }

    copy_item(config_dict, batch_spec, 'server')
    copy_item(config_dict, batch_spec, 'scheduler')
    copy_item(config_dict, batch_spec, 'backend')

    # 4. write to file
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', newline='\n') as f:
        f.write(json.dumps(batch_spec, indent=4))
    return batch_spec


def _load_batch_data_dir(batches_data_dir: str, batch_name) -> BatchApplication:
    batch_data_dir_path = Path(batches_data_dir) / batch_name
    spec_file_path = batch_data_dir_path / Batch.FILE_CONFIG
    if not spec_file_path.exists():
        raise RuntimeError(f"batch {batch_name} not exists")

    batch_spec_dict = load_json(spec_file_path)
    return BatchApplication.load(batch_spec_dict, batch_data_dir_path)


def run_show_jobs(batch_name, batches_data_dir):
    batch_app = _load_batch_data_dir(batches_data_dir, batch_name)
    batch = batch_app.batch
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    api_server_portal = batch_app.web_app.portal

    jobs_dict = api.list_jobs(api_server_portal)

    headers = ['name', 'status']
    tb = pt.PrettyTable(headers)
    for job_dict in jobs_dict:
        row = [job_dict.get(header) for header in headers]
        tb.add_row(row)
    print(tb)


def run_kill_job(batch_name, job_name, batches_data_dir):
    batch_app = _load_batch_data_dir(batches_data_dir, batch_name)
    batch = batch_app.batch
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    api_server_portal = batch_app.web_app.portal

    jobs_dict = api.kill_job(api_server_portal, job_name)
    print(json.dumps(jobs_dict))


def show_job(batch_name, job_name, batches_data_dir):
    batch_app = _load_batch_data_dir(batches_data_dir, batch_name)
    batch = batch_app.batch
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    api_server_portal = batch_app.web_app.portal

    job_dict = api.get_job(job_name, api_server_portal)
    job_desc = json.dumps(job_dict, indent=4)
    print(job_desc)


def run_batch_config_file(config, batches_data_dir):
    config_dict = load_json(config)
    run_batch_config(config_dict, batches_data_dir)


def run_show_batches(batches_data_dir):
    batches_data_dir_path = Path(batches_data_dir)

    if not batches_data_dir_path.exists():
        print("null")
        return

    # batches_data_dir_path = Path(batches_data_dir)
    # batches_name =
    batches_name = []
    batches_summary = []
    for filename in os.listdir(batches_data_dir):
        if (batches_data_dir_path / filename).is_dir():
            batches_name.append(filename)
            batch_app = _load_batch_data_dir(batches_data_dir, filename)

            batch_summary = batch_app.summary_batch()
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


def main():
    """
    Examples:
        cd hypernets/tests/hyperctl/
        hyperctl run --config ./local_batch.json
        hyperctl batch list
        hyperctl job list --batch-name=local-batch-example
        hyperctl job describe --job-name=job1 --batch-name=local-batch-example
        hyperctl job kill --job-name=job1 --batch-name=local-batch-example
        hyperctl job kill --job-name=job2 --batch-name=local-batch-example
        hyperctl batch list
    :return:
    """

    bdd_help = f"batches data dir, default get from environment variable {consts.KEY_ENV_BATCHES_DATA_DIR}"
    default_batches_data_dir = get_default_batches_data_dir()

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
        batch_list_parse.add_argument("--batches-data-dir", help=bdd_help,
                                      default=default_batches_data_dir, required=False)

    def setup_job_parser(operation_parser):

        exec_parser = operation_parser.add_parser("job", help="job operations")

        batch_subparsers = exec_parser.add_subparsers(dest="job_operation")

        job_list_parse = batch_subparsers.add_parser("list", help="list jobs")
        job_list_parse.add_argument("-b", "--batch-name", help="batch name", default=None, required=True)
        job_list_parse.add_argument("--batches-data-dir", help=bdd_help,
                                    default=default_batches_data_dir, required=False)

        def add_job_spec_args(parser_):
            parser_.add_argument("-b", "--batch-name", help="batch name", default=None, required=True)
            parser_.add_argument("-j", "--job-name", help="job name", default=None, required=True)
            parser_.add_argument("--batches-data-dir", help=bdd_help, default=default_batches_data_dir,
                                 required=False)

        job_kill_parse = batch_subparsers.add_parser("kill", help="kill job")
        add_job_spec_args(job_kill_parse)

        job_describe_parse = batch_subparsers.add_parser("describe", help="describe job")
        add_job_spec_args(job_describe_parse)

    def setup_run_parser(operation_parser):
        exec_parser = operation_parser.add_parser("run", help="run jobs")
        exec_parser.add_argument("-c", "--config", help="specific jobs json file", default=None, required=True)
        exec_parser.add_argument("--batches-data-dir", help=bdd_help, default=default_batches_data_dir,
                                 required=False)

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
        run_batch_config_file(**kwargs)
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

