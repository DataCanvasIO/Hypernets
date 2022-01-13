# hyperctl

hyperctl is used to manage jobs.

**Get started**

Install `hypernets` and type command `hyperctl` you shall see following outputs:
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

Create a python script `~/job.py` to run job with following content:
```python
from hypernets import hyperctl

params = hyperctl.get_job_params()
assert params
print(params)
```

put the minimum jobs' specification config into a file `jobs.json`:
```json
{
    "jobs": [
        {
            "params": {
                "learning_rate": 0.1
            },
            "execution": {
                "command": "python ~/job.py"
            }
        }
    ]
}
```

run the job with command:
```shell
hyperctl run --config ./jobs.json
```

**Generate jobs**

hyperctl can combine parameters to generate jobs in a batch. For example if your has several params and each params has 
multiple choices like :
```shell
{
  "learning_rate": [0.1, 0.2],
  "max_depth": [3, 5]
}
```
it shall generate jobs with params:
- `{ "learning_rate": 0.1, "max_depth": 3 }`
- `{ "learning_rate": 0.1, "max_depth": 5 }`
- `{ "learning_rate": 0.2, "max_depth": 3 }`
- `{ "learning_rate": 0.2, "max_depth": 5 }`

To batch generate jobs you should write job template file `jobs.yml` with following content:

```yaml
version: 2.5
params:
    learning_rate: [0.1,0.2]
    max_depth: [3, 5]
data_dir: /tmp/hyperctl-batches-data
execution:
  command: python3 main.py
backend:
  type: local
  conf:
```

run command to generate jobs specification config:
```shell
hyperctl generate --config ./jobs.yml --output ./jobs-generated.json
```

**Parallel jobs in remote machines**

You can run jobs in remote machines via SSH. Configure the host's connections setting in remote `backend` options:
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
}
```

**Job configuration's description**
```
{
    "jobs": [
        {
            "name": "aVqNV5Ut1",  // str, unique in batch, optional, if is null will generate a uuid as job name
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
                "command": "sleep 3", // str, required, command to the the job
                "working_dir": "/tmp", // str, optional, default is execution.data_dir,  working dir to run the command
                "data_dir": "/tmp/hyperctl-batch-data/aVqNV5Ut1"  // str, optional, default is <batch_data_dir>/<job_name>, the directory to write job's output data
            }
        }
    ],
    "backend": { // dict, optional, default is local backend, is where the jobs running on. 
        "type": "local", // str, one of "local,remote"
        "conf": {}  // 
    },
    "name": "eVqNV5Ut1",  // str, optional, batch name, default is a new uuid 
    "daemon": {  // dict, optional, daemon process setting
        "port": 8060,  // int, optional, default is 8060, http service port
        "exit_on_finish": false  // boolean, optional , default is false, whether to exit the process when all jobs are finished
    },
    "version": "2.5" // str, optional, default using the running hyperctl's version
}
```

