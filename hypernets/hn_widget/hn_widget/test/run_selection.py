from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from hn_widget import JupyterHyperModelCallback, JupyterWidgetExperimentCallback
from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypergbm.tests import test_output_dir
from hypernets.core.callbacks import EarlyStoppingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment import CompeteExperiment
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils

df = dsutils.load_bank()

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')

rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache',
              callbacks=[JupyterHyperModelCallback(), EarlyStoppingCallback(3, 'max', time_limit=60, expected_reward=1)])

ce = CompeteExperiment(hk, X_train, y_train,
                       scorer=get_scorer('roc_auc'),
                       feature_selection=True,
                       feature_selection_strategy='threshold',
                       feature_selection_threshold=100,
                       feature_selection_quantile=None,
                       feature_selection_number=None,
                       callbacks=[JupyterWidgetExperimentCallback()])
pipeline = ce.run(max_trails=3)

