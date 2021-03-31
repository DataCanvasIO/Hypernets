from hypernets.conf import configure, Configurable, String, Int, Float, Enum


@configure()
class DispatchCfg(Configurable):
    experiment = String(help='experiment id',
                        ).tag(config=True)
    work_dir = String(help='storage directory path to store running data.'
                      ).tag(config=True)
    backend = Enum(['standalone', 'dask', 'cluster', None],
                   default_value=None,
                   help='dispatcher backend'
                   ).tag(config=True)
    trial_retry_limit = Int(1000, min=1,
                            help='maximum retry number to run trial.'
                            ).tag(config=True)

    cluster_driver = String(help='driver address, used if backend="cluster"'
                            ).tag(config=True)
    cluster_role = Enum(['driver', 'executor'],
                        help='node role, used if backend="cluster"'
                        ).tag(config=True)
    cluster_search_queue = Int(1, min=1,
                               help='search queue size, used if backend="cluster"'
                               ).tag(config=True)
    cluster_summary_interval = Float(60.0,
                                     help='summary interval seconds',
                                     ).tag(config=True)

    dask_search_queue = Int(1, min=1,
                            help='search queue size, used if backend="dask"'
                            ).tag(config=True)
    dask_search_executors = Int(3, min=1,
                                help='search executor number, used if backend="dask"'
                                ).tag(config=True)

    grpc_worker_count = Int(10, min=1,
                            help='grpc worker count'
                            ).tag(config=True)
