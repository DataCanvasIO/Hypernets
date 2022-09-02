import os
from pathlib import Path

KEY_ENV_BATCHES_DATA_DIR = 'HYPERCTL_BATCHES_DATA_DIR'
KEY_ENV_JOB_NAME = 'HYPERCTL_JOB_NAME'
KEY_ENV_JOB_DATA_DIR = 'HYPERCTL_JOB_DATA_DIR'
KEY_ENV_JOB_WORKING_DIR = 'HYPERCTL_JOB_WORKING_DIR'
KEY_ENV_SERVER_PORTAL = 'HYPERCTL_SERVER_PORTAL'
KEY_ENV_TMP = 'TMP'

# placeholder
P_HOST_ENV = 'P_HOST_ENV'
P_TMP_ENV = 'P_TMP_ENV'

KEY_TEMPLATE_COMMAND = "COMMAND"

RUN_SH_TEMPLATE = f"""#!/bin/sh
export {KEY_ENV_JOB_NAME}="#{KEY_ENV_JOB_NAME}#"
export {KEY_ENV_JOB_DATA_DIR}="#{KEY_ENV_JOB_DATA_DIR}#"
export {KEY_ENV_SERVER_PORTAL}="#{KEY_ENV_SERVER_PORTAL}#"
export {KEY_ENV_JOB_WORKING_DIR}="#{KEY_ENV_JOB_WORKING_DIR}#"

#{P_TMP_ENV}
#{P_HOST_ENV}

if [ -n "$TMP" ]; then
  if [ ! -d "$TMP" ]; then
    mkdir -p $TMP
  fi
fi

cd ${KEY_ENV_JOB_WORKING_DIR}
#{KEY_TEMPLATE_COMMAND}# >"${KEY_ENV_JOB_DATA_DIR}/stdout" 2>"${KEY_ENV_JOB_DATA_DIR}/stderr"
"""

HOST_LOCALHOST = "localhost"

BATCH_TEMP = "hynctl_batch_temp"

JOB_DATA_DIR_PREFIX = "hynctl_job_"


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
