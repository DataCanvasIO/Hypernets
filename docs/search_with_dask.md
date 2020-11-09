## Use Dask to parallelize Hypernets search

In each HyperModel `search()` call, 
it loops the HyperSpace `sample()` -> the Estimator `fit()` and 
`evaluate()` until the `max_trails` reached or the`EarlyStoppingError`
occurred. Hypernets executes the loop step by step within one
python process by default.

Usually, the Estimator `fit()` and `evaluate()` spend most
CPU time. Hypernets provides the ability to parallelize Estimator
`fit()` and `evaluate()` with the [Dask](https://dask.org/) cluster. 

To turn Dask support on, please do the following steps:
1. Setup Dask local cluster or distributed cluster, 
you can refer to the [Dask setup documents](https://docs.dask.org/en/latest/setup.html).
1. Set Hypernets environment variable `HYN_SEARCH_BACKEND` to `dask`
1. Set Dask environment variable `DASK_SCHEDULER_ADDRESS`
to dask scheduler address,
or initialize Dask Client object before run HyperModel `search()`.
1. Optional, you can set Hypernets environment variable
`HYN_SEARCH_EXECUTORS` to control the parallelized executor count, 
default is `3`.

