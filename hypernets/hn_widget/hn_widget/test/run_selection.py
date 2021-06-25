
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
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')

rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache',
              callbacks=[JupyterHyperModelCallback(), EarlyStoppingCallback(3, 'max', time_limit=60, expected_reward=1)])

from hypernets.core.nb_callbacks import JupyterWidgetExperimentCallback
ce = CompeteExperiment(hk, X_train, y_train,
                    scorer=get_scorer('roc_auc'),
                    feature_selection=False,
                    feature_selection_strategy='threshold',
                    feature_selection_threshold=0.0001,
                    feature_selection_quantile=None,
                    feature_selection_number=None,
                    callbacks=[JupyterWidgetExperimentCallback()])
pipeline = ce.run(max_trails=3)

y_score = pipeline.predict_proba(X_test)

import pickle as pkl
with open('/tmp/y_score.pkl', 'wb') as f:
    pkl.dump(y_score, f)


