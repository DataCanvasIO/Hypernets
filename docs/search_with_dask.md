## Use Dask to parallelize Hypernets search

In each HyperModel `search()` call, 
it loops the HyperSpace `sample()` -> HyperModel `_run_trail()`
until the `max_trails` reached or the`EarlyStoppingError`
occurred. Hypernets executes the loop step by step within one
python process by default.

Usually, the HyperModel `_run_trail()` which execution the esspend most CPU time. 
Hypernets provides the ability to parallelize `_run_trail()`
with the [Dask](https://dask.org/) cluster. 

### Setup Dask support

To turn Dask support on, please do the followings:
1. Setup Dask cluster, refer to the
 [Dask setup documents](https://docs.dask.org/en/latest/setup.html) pls.
1. Set Dask environment variable `DASK_SCHEDULER_ADDRESS`
 to dask scheduler address, or initialize Dask `Client` instance
 before run HyperModel `search()`.
1. Set Hypernets environment variable `HYN_SEARCH_BACKEND` to `dask`
1. Optional, set Hypernets environment variable
`HYN_SEARCH_EXECUTORS` to control the concurrent Dask job number, 
default is `3`.

### Setup share storage for distributed Dask cluster

If you plan setup a distributed Dask cluster with multiple nodes
to run Hypernets search, a share storage is required. 
Currently, Hypernets can use shared file system 
or S3-based object store to share data between 
the Hypernets main process and Dask jobs.

#### Use shared file system

1. Deploy your share file system, such as NFS, cephfs, etc.
1. Mount the share file system in Hypernets node and all Dask node, 
with the same access path, eg: `/opt/share`, 
and grant read-write permissions to the user which starts 
Hypernets and Dask cluster.
1. Set the following Hypernets environment variables:
    * `HYN_STORAGE_TYPE`: set to `file`
    * `HYN_STORAGE_ROOT`: set to the shared storage root path for Hypernets,
    eg: `/opt/share/hypernets`


####  Use S3-base object store

1. Request S3 compatible service from your cloud provider,
or deploy your private object store service use [ceph](https://ceph.io/),
 [minio](https://min.io/), etc.
1. Set the following Hypernets environment variables:
    * `HYN_STORAGE_TYPE`: set to `s3`
    * `HYN_STORAGE_ROOT`: set to root path for Hypernets, 
    eg: `/bucket_name/some_paths`
    * `HYN_STORAGE_OPTIONS`: set a connection string with json format
     to tell Hypernets how to access your object store service, 
     eg: `{"anon":false,"client_kwargs":{"endpoint_url":"your_service_address","aws_access_key_id":"...","aws_secret_access_key":"..."}}`.
     Refer to [s3fs](https://s3fs.readthedocs.io/en/latest/)
     for more connection information.
     

