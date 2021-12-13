from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general

from hypernets.core.searcher import OptimizeDirection

from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils
from hypergbm.tests import test_output_dir
from hypernets.experiment import CompeteExperiment

from hn_widget import JupyterHyperModelCallback, JupyterWidgetExperimentCallback
from hypernets.core.callbacks import EarlyStoppingCallback

df = dsutils.load_bank()

X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')

rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache',
              callbacks=[JupyterHyperModelCallback(), EarlyStoppingCallback(10, 'max', time_limit=60,
                                                                            expected_reward=1)])
ce = CompeteExperiment(hk, X_train, y_train, callbacks=[JupyterWidgetExperimentCallback()])
ce.run(max_trials=3)

print(ce)
