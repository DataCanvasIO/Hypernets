from hypernets.utils.logging import set_level
set_level(40)
from hypernets.utils import logging as hn_logging

from sklearn.model_selection import train_test_split
from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment.compete import EnsembleStep
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils
from hypergbm.tests import test_output_dir
from hypernets.experiment.compete import CompeteExperiment
import json
from hypernets.core.nb_callbacks import JupyterWidgetExperimentCallback
from hypernets.core.nb_callbacks import JupyterHyperModelCallback
from sklearn.metrics import get_scorer
from hypernets.core.callbacks import EarlyStoppingCallback
import dask.dataframe as dd

df = dsutils.load_bank()

def to_dask(_df):
    return dd.from_pandas(_df, npartitions=2)

# df.drop(['id'], axis=1, inplace=True)
X_train, X_test = train_test_split(df.head(3000), test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')
X_prediction = df.sample(3000, random_state=66)
X_prediction.drop('y', inplace=True, axis=1)

X_train = to_dask(X_train)
X_test = to_dask(X_test)
y_train = to_dask(y_train)
y_test = to_dask(y_test)
X_prediction = to_dask(X_prediction)

rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache',
              callbacks=[JupyterHyperModelCallback(), EarlyStoppingCallback(30, 'max', time_limit=18000, expected_reward=1)])


ce = CompeteExperiment(hk, X_train, y_train,
                       X_test=X_prediction,
                       pseudo_labeling=True,
                       pseudo_labeling_strategy='number',
                       pseudo_labeling_proba_threshold=0.8,
                       pseudo_labeling_proba_quantile=0.3,
                       pseudo_labeling_sample_number=10,
                       pseudo_labeling_resplit=False,
                       drift_detection=True,
                       drift_detection_remove_shift_variable=False,
                       drift_detection_variable_shift_threshold=0.51,
                       drift_detection_threshold=0.47,
                       drift_detection_min_features=6,
                       drift_detection_remove_size=0.2,
                       collinearity_detection=False,
                       feature_generation=False,
                       feature_generation_trans_primitives=["cross_categorical", "add_numeric", "subtract_numeric"],
                       feature_generation_max_depth=1,
                       feature_generation_categories_cols=['job', 'education'],
                       feature_generation_continuous_cols=['balance', 'duration'],
                       feature_generation_datetime_cols=None,
                       feature_generation_latlong_cols=None,
                       feature_generation_text_cols=None,
                       scorer=get_scorer('roc_auc'),
                       feature_selection=False,
                       feature_selection_strategy='threshold',
                       feature_selection_threshold=0.001,
                       feature_selection_quantile=None,
                       feature_selection_number=None,
                       feature_reselection=False,
                       feature_reselection_estimator_size=10,
                       feature_reselection_strategy='threshold',
                       feature_reselection_threshold=0.1,
                       feature_reselection_quantile=None,
                       feature_reselection_number=None,
                       callbacks=[JupyterWidgetExperimentCallback()],
                       log_level=hn_logging.ERROR
                      )
print(ce)
from hn_widget import experiment_util

d = experiment_util.extract_experiment(ce)
print(json.dumps(d))
