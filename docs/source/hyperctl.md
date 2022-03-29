# Hyperctl

Hyperctl is a general tool for multi-job management, which includes but not limit to training, testing and comparison. It is packaged under Hypernets and intended to provide convenience to every developing stage. 


**Get started**

After installing `hypernets`, you could see the following description by typing `hyperctl`, which includes four arguments  `run`, `generate`, `batch`, `job':
```shell

$ hyperctl
usage: hyperctl [-h] {run,generate,batch,job} ...

hyperctl command is used to manage jobs

positional arguments:
  {run,generate,batch,job}
    run                 run jobs
    generate            generate specific jobs json file
    batch               batch operations
    job                 job operations

optional arguments:
  -h, --help            show this help message and exit

```

**Examples**

EX1: Single job execution

- First, create a job python script `~/job-script.py`:
```python
from hypernets import hyperctl

params = hyperctl.get_job_params()
assert params
print(params)
```

- You could get a list of parameters. Select the parameters you would like to configure and filled in the minimum jobs' specification config (e.g., `learning_rate`). Then you could get the `batch.json` file:
```json
{
    "jobs": [
        {
            "params": {
                "learning_rate": 0.1
            },
            "execution": {
                "command": "python ~/plain_job_script.py"
            }
        }
    ]
}
```

- Run the job with command:
```shell
hyperctl run --config ./batch.json
```

EX2. Generate a series multi-job 'batch.json' file

When existing multiple parameters with multiple choices, Hyperctl can generate a multi-job file with all permutations. For instance, there are two parameters `"learning rate":[0.1, 0.2]` and `"max_depth": [3,5]`. The total permutations of parameter configurations is four. Hyperctl provides a `job-template.yml` file, from which it could automatically generate the 'batch.json' file.

The `job-template.yml` includes the following contents.
```
name: eVqNV5Ut1   // the same as job configuration `name`
version: "2.5"    // the same as job configuration `version`
params:           // dict[str, list], required, value should be list 
    param1: ["value1", "value2"]
execution:        // the same as job configuration `jobs.execution`
resource:         // the same as job configuration `jobs.resource`
daemon:           // the same as job configuration `daemon`
backend:          // the same as job configuration `backend`
```

Below is the example of how to generate the 'job-template.yml' and 'batch.json' files with two parameters:

```yaml
params:
    learning_rate: [0.1,0.2]
    max_depth: [3, 5]
execution:
  command: python3 cli.py
```

```shell
hyperctl generate --template ./job-template.yml --output ./batch.json
```


EX3. Parallel jobs in remote machines

Hyperctl also supports parallel jobs in remote machines by configuring the argument `backend`. An example of the running in remote machines via SSH in shown below.

```
{
 "backend": {
        "type": "remote",
        "conf": {
            "machines": [  // list, required, specific remote machines's SSH connection setting 
                {
                    "hostname": "host1", // str, required
                    "username": "hyperctl", // str, required
                    "password": "hyperctl"  // str, optional if ssh_rsa_file is not null
                    "ssh_rsa_file": "~/.ssh/id_rsa" // str, optional if password is not null
                }
            ]
        }
    }
  "daemon": {
      "host": "192.168.10.206",  // str, optional, http service host ip, you should use IP address that can be accessed in remote machines
  }
}
```

Note that the configuration item `daemon.host` should be accessed by remote machines declared in the configuration `backend.conf.machines`,
Otherwise, the task will fail because the daemon server cannot be accessed.


EX4. Full job configuration

The example below shows a full job configuration.
```
{
    "name": "eVqNV5Ut1",  // str, optional, batch name, default is a new uuid, recommended that you specify one, with the specified {jobs.name} , the executed job can be skipped when the batch is re-executed
    "jobs": [
        {
            "name": "aVqNV5Ut1",  // str, unique in batch, optional, if is null will generate a uuid as job name, recommended that you specify one, with the name of the batch name, the executed job can be skipped when the batch is re-executed
            "params": {  // dict, required
                "learning_rate": 0.1,
                "dataset": "path/d1.csv"
            },
            "resource": {  // not available now, alloc a whole machine for a job every time
                "cpu": 2,
                "ram": 1024,
                "gpu": 1
            },
            "execution": {
                "command": "sleep 3", // str, required, command to the the job, if execute a file, recommend use absolute path or path relative to {execution.working_dir}
                "working_dir": "/tmp", // str, optional, default is execution.data_dir,  working dir to run the command
                "data_dir": "/tmp/hyperctl-batch-data/aVqNV5Ut1"  // str, optional, default is {batch_data_dir}/{job_name}, the directory to write job's output data
            }
        }
    ],
    "backend": { // dict, optional, default is local backend, is where the jobs running on. 
        "type": "local", // str, one of "local,remote"
        "conf": {}  // 
    },
    "daemon": {  // dict, optional, daemon process setting
        "host": "192.168.10.206",  // str, optional, default is localhosh, http service host ip, you should use IP address that can be accessed in remote machines if is remote backend
        "port": 8060,  // int, optional, default is 8060, http service port
        "exit_on_finish": false  // boolean, optional , default is false, whether to exit the process when all jobs are finished
    },
    "version": "2.5" // str, optional, default using the running hyperctl's version
}
```
