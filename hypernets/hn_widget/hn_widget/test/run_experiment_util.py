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
X_train['age_1'] = X_train['age']
y_train = X_train.pop('y')
y_test = X_test.pop('y')
X_prediction = df.sample(2000)
X_prediction['age_1'] = X_prediction['age']
rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)

hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache')

ce = CompeteExperiment(hk, X_train, y_train,
                       X_test=X_prediction,
                       drift_detection_variable_shift_threshold=0.51,
                       drift_detection_threshold=0.7,
                       drift_detection_min_features=10,
                        drift_detection_remove_size=0.7,
                        collinearity_detection=True,
                       callbacks=[JupyterWidgetExperimentCallback()])
print(ce)

# ce = CompeteExperiment(hk, X_train, y_train,
#                        X_test=X_prediction,
#                        drift_detection_variable_shift_threshold=0.51,
#                        drift_detection_threshold=0.46,
#                        drift_detection_min_features=3,
#                         drift_detection_remove_size=0.5,
#                         collinearity_detection=True,
#                        callbacks=[JupyterWidgetExperimentCallback()])

from hn_widget import experiment_util

steps_dict = experiment_util.extract_experiment(ce)
print(steps_dict)

s = ce.run(max_trials=2)
print(s)
steps_dict = experiment_util.extract_experiment(ce)
print(steps_dict)
print(steps_dict)
