=========
Hyperctl
=========

Hyperctl is a general tool for multi-job management, which includes but not limit to training, testing and comparison. It is packaged under Hypernets and intended to provide convenience to every developing stage. 


Concepts
=============

**Job**

A command line job that accepts parameters.
Hyperctl provides the python API to read the parameters of the job, so this command line can execute a python script and use the API to obtain the parameters to complete the job.


**Batch**

A batch of jobs. All the status files and output files of jobs in the same batch are in the working directory of the batch.

**Scheduler**

Job scheduler, which schedules jobs in batch to run on appropriate machine resources and manages computing resources.

**Backend**

The backend for running jobs can run in a stand-alone mode or multiple remote nodes through SSH protocol.


Examples
================

Run batch with command line tool
__________

After installing ``hypernets``, you could see the following description by typing ``hyperctl``, which includes four arguments  ``run``, ``generate``, ``batch``, ``job``:

.. code-block:: shell

    $ hyperctl
    usage: hyperctl [-h] [--log-level LOG_LEVEL] [-error] [-warn] [-info] [-debug] {run,generate,batch,job} ...

    hyperctl command is used to manage jobs

    positional arguments:
      {run,generate,batch,job}
        run                 run jobs
        generate            generate specific jobs json file
        batch               batch operations
        job                 job operations

    optional arguments:
      -h, --help            show this help message and exit

    Console outputs:
      --log-level LOG_LEVEL
                            logging level, default is INFO
      -error                alias of "--log-level=ERROR"
      -warn                 alias of "--log-level=WARN"
      -info                 alias of "--log-level=INFO"
      -debug                alias of "--log-level=DEBUG"

    ```

Take using Hyperctl to tuning the ``tol`` parameter of ``sklearn.linear_model.LogisticRegression`` as an example,
create a job python script ``~/sklearn_iris_example.py`` with following content:

.. code-block:: python

    from hypernets import hyperctl
    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    job_params = hyperctl.get_job_params()  # read job params as a dict from hyperctl

    tol=job_params['tol']

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8086)

    lr = LogisticRegression(tol=tol)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print(f"tol: {tol}, accuracy_score: {accuracy_score(y_pred, y_test)}")


Hyperctl uses the JSON format file to define the jobs, for example create a file named ``batch.json`` and configures 2 jobs then set the parameter ``tol`` to 1 and 100 respectively:

.. code-block:: python

    {
        "name": "sklearn_iris_example",
        "jobs": [{
                "name": "tol_1",
                "params": {
                    "tol": 1
                },
                "command": "python ~/sklearn_iris_example.py"
            },
            {
                "name": "tol_100",
                "params": {
                    "tol": 100
                },
                "command": "python ~/sklearn_iris_example.py"
            }
        ]
    }

.. note::

  Make sure that the python used by the command in the job has ``scikit-learn`` installed


Run the job with command:

.. code-block:: shell

    $ hyperctl run --config ./batch.json


After the task finished, view the output log file:

.. code-block:: shell

    ~/hyperctl-batches-working-dir/sklearn_iris_example/tol_1/stdout
    ----------------------------------------------------------------
    tol: 1, accuracy_score: 0.9333333333333333

.. code-block:: shell

    ~/hyperctl-batches-working-dir/sklearn_iris_example/tol_100/stdout
    ------------------------------------------------------------------
    tol: 100, accuracy_score: 0.36666666666666664


Run batch with API
_________________________________




Generate batch config from job template
_______________________________________

Hyperctl generates jobs config in batch by arranging and combining parameters based on the configuration template, the generated file can be used to run the batch.
Here is an example of how to use template file to generate batch config file .
First create a template file ``job-template.yml`` with following content:

.. code-block:: yaml

    params:
        learning_rate: [0.1,0.2]
        max_depth: [3, 5]
    command: python3 cli.py



Then execute command to generate batch config file:

.. code-block:: shell

    $ hyperctl generate --template ./job-template.yml --output ./batch.json


Here is the generated ``batch.json`` file:

.. code-block:: json

    {
        "name": "eVqNV5Ut1",
        "job": [{
            "name": "eaqNV5Ut1",
            "params": {
                "learning_rate": 0.1,
                "max_depth": 3
            },
            "command": "python3 cli.py"
        }, {
            "name": "ebqNV5Ut1",
            "params": {
                "learning_rate": 0.1,
                "max_depth": 5
            },
            "command": "python3 cli.py"
        }, {
            "name": "ecqNV5Ut1",
            "params": {
                "learning_rate": 0.2,
                "max_depth": 3
            },
            "command": "python3 cli.py"
        }, {
            "name": "edqNV5Ut1",
            "params": {
                "learning_rate": 0.2,
                "max_depth": 5
            },
            "command": "python3 cli.py"
        }]
    }




Batch configuration file references
====================================


Examples
__________

LocalBackend
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "name": "local_backend_example",
        "jobs": [
            {
                "name": "job1",
                "params": {
                    "param1": 1
                },
                "command": "sleep 3"
            }
        ],
        "backend": {
            "type": "local"
        }
    }

RemoteSSHBackend
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "name": "local_backend_example",
        "jobs": [
            {
                "name": "job1",
                "params": {
                    "param1": 1
                },
                "command": "sleep 3"
            }
        ],
        "backend": {
            "type": "remote",
            "machines": [
                {
                    "connection": {
                        "hostname": "host1",
                        "username": "hyperctl",
                        "password": "hyperctl"
                    }
                }
            ]
        },
        "server": {
          "host": "192.168.10.206"
        }
    }


Configuration references
________________________

BatchApplicationConfig
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - name
      - ``str``, required
      - batch name, should be unique in a batch.

    * - jobs
      - list[`JobConfig`_], required
      - Jobs to run.

    * - backend
      - `BackendConfig`_, optional
      -  platform where the jobs running on, default is `LocalBackendConfig`_ .

    * - server
      - `ServerConfig`_ , optional
      - server setting.

    * - scheduler
      - `SchedulerConfig`_ , optional
      -  scheduler setting.

    * - batches_data_dir
      - ``str``, optional
      - batches working directory, where to store output files of batches, hyperctl will create a sub-directory by the batch name for every batch in this directory.
        default read from environment by key ``HYPERCTL_BATCHES_DATA_DIR``, if do not set in environments using ``~/hyperctl-batches-data-dir``.

    * - version
      - ``str``, optional
      - if is None, use the currently running version, default is None.


JobConfig
^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - name
      - ``str``, optional
      - str, unique in batch, optional, if is null will generate a uuid as job name, recommended that you specify one, with the name of the batch name, the executed job can be skipped when the batch is re-executed

    * - params
      - ``dict``, required
      - job params, it can be obtained through API ``hypernets.hyperctl.get_job_params``

    * - command
      - ``str``, required
      - command to the the job, if execute a file, recommend use absolute path or path relative to {execution.working_dir}

    * - working_dir
      - ``str``, optional
      -  working dir to run the ``command``, default is {batches_data_dir}/{batch_name}/{job_name}


.. note::

  A job write output file to ``{batches_data_dir}/{batch_name}/{job_name}``, it usually contains files:

    - stdout: standard output
    - stderr: standard error
    - run.sh: shell script to run the job


BackendConfig
^^^^^^^^^^^^^^^^^^^^^^

Is one of :

- `LocalBackendConfig`_
- `RemoteBackendConfig`_


LocalBackendConfig
^^^^^^^^^^^^^^^^^^^^^^

Running batch in standalone mode,  please refer to the example `LocalBackend`_.

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - type
      - ``"local"``
      -

    * - environments
      - ``dict``, optional
      - Environments setting will export for the job process.


RemoteBackendConfig
^^^^^^^^^^^^^^^^^^^^^^

Hyperctl supports parallel jobs in remote machines, this mode uses multiple machines to speed up the progress of the batch.
It distributes jobs to remote nodes through the SSH protocol, which requires that the nodes running tasks remotely need to run SSH services and provide connection accounts.
Please refer to the example `RemoteSSHBackend`_ 。

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - machines
      - list[`RemoteMachineConfig`_ ], required
      - Connection and configuration information of remote machines.


RemoteMachineConfig
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - connection
      - `SHHConnectionConfig`_, required
      - Connection information for the remote machine.

    * - environments
      - ``dict``, optional
      - Environments setting will export for the job process.


SHHConnectionConfig
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - hostname
      - ``hostname``, required
      - IP or hostname of remote machine.

    * - username
      - ``username``, required
      - username of remote machine.

    * - password
      - ``password``, required
      - password of remote machine.


ServerConfig
^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - host
      - ``str``, optional
      - where to bind for the http server, it's should be IP address that can be accessed in remote machines if is remote backend, otherwise, the job will fail because the api server cannot be accessed, default is localhost.

    * - port
      - ``int``, optional
      - http server port, default is 8060


SchedulerConfig
^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - interval
      - ``int``, optional
      - Scheduling interval, the unit is milliseconds, default value is 5000

    * - exit_on_finish
      - ``boolean``, optional
      -  whether to exit the process when all jobs are finished, default is false


Job template configuration file references
===========================================

Examples
________

Basic example
^^^^^^^^^^^^^^^^^^^^^^

Refer to `Job template`_ .

Configuration references
_________________________

JobTemplateConfig
^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Field Name
      - Type
      - Description

    * - name
      - ``str``, required
      - refer to ``BatchApplicationConfig.name``

    * - params
      - ``dict[str, list]``, required
      - job params list, used to arrange and combine to generate jobs config.

    * - command
      - ``str``, required
      - refer to ``JobConfig.command``

    * - working_dir
      - ``dict[str, list]``, required
      -
    * - backend
      - `BackendConfig`_, optional
      - refer to ``BatchApplicationConfig.backend``

    * - batches_data_dir
      -  ``str``, optional
      - refer to ``BatchApplicationConfig.batches_data_dir``

    * - server
      - `ServerConfig`_ , optional
      - refer to ``BatchApplicationConfig.server``

    * - scheduler
      - `SchedulerConfig`_ , optional
      -  refer to ``BatchApplicationConfig.scheduler``

    * - version
      - ``str``, optional
      - refer to ``BatchApplicationConfig.version``
