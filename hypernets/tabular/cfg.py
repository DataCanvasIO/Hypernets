from hypernets.conf import configure, Configurable, Int, String, Enum


@configure()
class TabularCfg(Configurable):
    joblib_njobs = \
        Int(-1, allow_none=True,
            help='"n_jobs" setting for joblib task.'
            ).tag(config=True)

    permutation_importance_sample_limit = \
        Int(10000, min=100,
            help='maximum number to run permutation importance.'
            ).tag(config=True)

    cache_strategy = \
        Enum(['data', 'transform', 'disabled'],
             default_value='transform',
             config=True,
             help='dispatcher backend',
             )

    cache_dir = \
        String('cache_dir',
               allow_none=False,
               config=True,
               help='the directory to store cached data, read/write permissions are required.')

    geohash_precision = \
        Int(12,
            config=True,
            help=''
            )
