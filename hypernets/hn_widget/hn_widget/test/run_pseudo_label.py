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
from hypernets.core.nb_callbacks import JupyterWidgetExperimentCallback


df = dsutils.load_bank()
# df.drop(['id'], axis=1, inplace=True)
X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')
X_prediction = df.sample(2000, random_state=66)
X_prediction.drop('y', inplace=True, axis=1)

rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache')


ce = CompeteExperiment(hk, X_train, y_train,
                       X_test=X_prediction,
                       collinearity_detection=False,
                       drift_detection=False,
                       pseudo_labeling=True,
                       pseudo_labeling_strategy='threshold',
                       pseudo_labeling_proba_threshold=0.6,
                       pseudo_labeling_proba_quantile=None,
                       pseudo_labeling_sample_number=None,
                       pseudo_labeling_resplit=False,
                       callbacks=[JupyterWidgetExperimentCallback()])

from hn_widget import experiment_util
pipeline = ce.run(max_trials=2)
steps_dict = experiment_util.extract_experiment(ce)
import json
print(json.dumps(steps_dict))


