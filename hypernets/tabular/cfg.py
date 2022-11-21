import os
import sys as sys_

from hypernets.conf import configure, Configurable, Int, String, Bool, Float, Enum, Dict

if sys_.platform.find('win') == 0:
    # _joblib_default_options = dict(backend='multiprocessing')
    _joblib_default_options = dict(prefer='processes')
else:
    _joblib_default_options = dict(prefer='processes')


def _safe_int(v, default=0):
    try:
        return int(v)
    except:
        return default


@configure()
class TabularCfg(Configurable):
    joblib_njobs = \
        Int(_safe_int(os.environ.get('OMP_NUM_THREADS', None), -1), allow_none=True,
            help='"n_jobs" setting for joblib task.'
            ).tag(config=True)

    joblib_options = \
        Dict(default_value=_joblib_default_options,
             allow_none=False,
             key_trait=String(),
             help='parallel settings except "n_jobs" setting for joblib task.'
             ).tag(config=True)

    shift_variable_sample_limit = \
        Int(10000, min=100,
            help='maximum number to run shift_variable detection.'
            ).tag(config=True)

    multi_collinearity_sample_limit = \
        Int(10000, min=100,
            help='maximum number to run multi collinearity.'
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
        Int(12, min=2,
            config=True,
            help=''
            )

    auto_categorize = \
        Bool(False,
             config=True,
             help=''
             )

    auto_categorize_shape_exponent = \
        Float(0.5,
              config=True,
              help=''
              )

    idness_threshold = \
        Float(0.99,
              config=True,
              help=''
              )

    column_selector_text_word_count_threshold = \
        Int(10, min=1,
            config=True,
            help=''
            )

    tfidf_max_feature_count = \
        Int(1000, min=2,
            config=True,
            help=''
            )

    tfidf_primitive_output_feature_count = \
        Int(30, min=2,
            config=True,
            help=''
            )
