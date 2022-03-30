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
from hypernets.hyperctl.batch import Batch, load_batch
from hypernets.hyperctl.scheduler import get_batches_data_dir  # FIXME
from hypernets.hyperctl.utils import load_yaml, load_json, copy_item
from hypernets.utils import logging as hyn_logging, common as common_util


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
        copy_item(config_dict, job_dict, 'resource')
        return job_dict

    jobs = [make_job_dict(_) for _ in itertools.product(*param_values)]

    # 3. merge to bath spec
    batch_spec = {
        "jobs": jobs,
        'name': config_dict.get('name', common_util.generate_short_id()),
        "version": config_dict.get('version', current_version)
    }

    copy_item(config_dict, batch_spec, 'backend')
    copy_item(config_dict, batch_spec, 'daemon')

    # 4. write to file
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', newline='\n') as f:
        f.write(json.dumps(batch_spec, indent=4))
    return batch_spec


def run_batch_from_config_file(config, batches_data_dir=None):
    config_dict = load_json(config)
    batch = prepare_batch(config_dict, batches_data_dir)
    _start_api_server(batch)


def load_batch_data_dir(batches_data_dir, batch_name):
    spec_file_path = Path(batches_data_dir) / batch_name / Batch.FILE_SPEC
    if not spec_file_path.exists():
        raise RuntimeError(f"batch {batch_name} not exists ")
    batch_spec_dict = load_json(spec_file_path)
    return load_batch(batch_spec_dict, batches_data_dir)


def run_show_jobs(batch_name, batches_data_dir=None):
    batches_data_dir = Path(get_batches_data_dir(batches_data_dir))
    batch = load_batch_data_dir(batches_data_dir, batch_name)
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    daemon_portal = batch.server_conf.portal

    jobs_dict = api.list_jobs(daemon_portal)

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

    daemon_portal = batch.server_conf.portal

    jobs_dict = api.kill_job(daemon_portal, job_name)
    print("Killed")


def show_job(batch_name, job_name, batches_data_dir=None):
    batches_data_dir = Path(get_batches_data_dir(batches_data_dir))
    batch = load_batch_data_dir(batches_data_dir, batch_name)
    if batch.STATUS_RUNNING != batch.status():
        raise RuntimeError("batch is not running")

    daemon_portal = batch.server_conf.portal

    job_dict = api.get_job(job_name, daemon_portal)
    job_desc = json.dumps(job_dict, indent=4)
    print(job_desc)


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
