import os
from pathlib import Path

KEY_DATA_DIR = 'data_dir'

KEY_ENV_BATCHES_DATA_DIR = 'HYPERCTL_BATCHES_DATA_DIR'
KEY_ENV_JOB_NAME = 'HYPERCTL_JOB_NAME'
KEY_ENV_JOB_DATA_DIR = 'HYPERCTL_JOB_DATA_DIR'
KEY_ENV_JOB_EXECUTION_WORKING_DIR = 'HYPERCTL_JOB_EXECUTION_WORKING_DIR'
KEY_ENV_DAEMON_PORTAL = 'HYPERCTL_DAEMON_PORTAL'

KEY_TEMPLATE_COMMAND = "COMMAND"

RUN_SH_TEMPLATE = f"""#!/bin/sh
export {KEY_ENV_JOB_NAME}="#{KEY_ENV_JOB_NAME}#"
export {KEY_ENV_JOB_DATA_DIR}="#{KEY_ENV_JOB_DATA_DIR}#"
export {KEY_ENV_DAEMON_PORTAL}="#{KEY_ENV_DAEMON_PORTAL}#"
export {KEY_ENV_JOB_EXECUTION_WORKING_DIR}="#{KEY_ENV_JOB_EXECUTION_WORKING_DIR}#"
cd ${KEY_ENV_JOB_EXECUTION_WORKING_DIR}
#{KEY_TEMPLATE_COMMAND}# >"${KEY_ENV_JOB_DATA_DIR}/stdout" 2>"${KEY_ENV_JOB_DATA_DIR}/stderr"
"""

HOST_LOCALHOST = "localhost"


def default_daemon_conf():
    return {
        "port": 8060,
        "exit_on_finish": False,
        "host": HOST_LOCALHOST
    }


def default_batches_data_dir(batches_data_dir):
    if batches_data_dir is None:
        bdd_env = os.environ.get(KEY_ENV_BATCHES_DATA_DIR)
        if bdd_env is None:
            bdd_default = Path("~/hyperctl-batches-data-dir").expanduser().as_posix()
            return bdd_default
        else:
            return bdd_env
    else:
        return batches_data_dir
