from hypernets.conf import configure, Configurable, Int

from collections import namedtuple
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
