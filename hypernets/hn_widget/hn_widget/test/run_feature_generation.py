from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment.compete import EnsembleStep
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils
from hypergbm.tests import test_output_dir
from hypergbm.experiment import CompeteExperiment
import json
from hypernets.core.nb_callbacks import JupyterHyperModelCallback
from hypernets.core.callbacks import EarlyStoppingCallback
from sklearn.metrics import get_scorer

df = dsutils.load_bank()
# df.drop(['id'], axis=1, inplace=True)
X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')

rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache',
              callbacks=[JupyterHyperModelCallback(), EarlyStoppingCallback(3, 'max', time_limit=60, expected_reward=1)])
from hypernets.core.nb_callbacks import JupyterWidgetExperimentCallback
ce = CompeteExperiment(hk, X_train, y_train,
                       feature_generation=True,
                       feature_generation_trans_primitives=["cross_categorical",  "add_numeric", "subtract_numeric"],
                       # feature_generation_fix_input=False,
                       feature_generation_max_depth=1,
                       feature_generation_categories_cols=['job', 'education'],
                       feature_generation_continuous_cols=['balance', 'duration'],
                       feature_generation_datetime_cols=None,
                       feature_generation_latlong_cols=None,
                       feature_generation_text_cols=None,
                    callbacks=[JupyterWidgetExperimentCallback()])
ce.run(max_trails=3)
print(ce)

