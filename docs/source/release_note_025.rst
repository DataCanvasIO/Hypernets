Version 0.2.5
-------------

We add a few new features to this version:

* Toolbox: A general computing layer for tabular data
  - Provide implementations of pandas, dask and cudf data types 
    - DefaultToolbox (Numpy + Pandas + Sklearn)
    - DaskToolbox (DaskCore + DaskML)
    - CumlToolBox (Cupy + Cudf + Cuml)


* HyperCtl: A tool package for multi-job management
  - Support sequencial jobs with multi-parameter settings
  - Support parallel jobs in remote multi-machines
 
 
* Export experiment report (.xlsx)
  - Include information of engineering features, ensembled models, evaluation scores, resource usages, etc.
  - Generate plots automatically 
