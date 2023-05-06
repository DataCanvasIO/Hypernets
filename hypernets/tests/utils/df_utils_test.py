from hypernets.utils import df_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder

from hypernets.experiment import CompeteExperiment, Experiment
from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from hypernets.tests.model.plain_model_test import create_plain_model
from hypernets.tests.tabular.tb_dask import if_dask_ready, is_dask_installed, setup_dask


def test_as_array():
    pd_df = dsutils.load_bank()
    pd_series = pd_df['id']

    assert isinstance(df_utils.as_array(pd_series), np.ndarray)
    assert isinstance(df_utils.as_array(pd_series.values), np.ndarray)
    assert isinstance(df_utils.as_array(pd_series.values.tolist()), np.ndarray)

    installed_cudf = False
    try:
        import cudf
        import cupy
        installed_cudf = True
    except Exception as e:
        pass

    if installed_cudf:
        import cudf

        cudf_series = cudf.from_pandas(pd_df)['id']
        assert isinstance(df_utils.as_array(cudf_series), np.ndarray)
        assert isinstance(df_utils.as_array(cudf_series.values), np.ndarray)


class Test_Get_character:
    @classmethod
    def setup_class(cls):
        if is_dask_installed:
            import dask.dataframe as dd
            setup_dask(cls)

            cls.boston = dd.from_pandas(dsutils.load_boston(), npartitions=1)
            cls.blood = dd.from_pandas(dsutils.load_blood(), npartitions=1)
            cls.bike_sharing = dd.from_pandas(dsutils.load_Bike_Sharing(), npartitions=1)

    # A test for multiclass task
    def experiment_with_bike_sharing(self, init_kwargs, run_kwargs, row_count=3000, with_dask=False):
        if with_dask:
            X = self.bike_sharing.copy()
            y = X.pop('count')
            y = y.astype('str')
        else:
            X = dsutils.load_Bike_Sharing()
            if row_count is not None:
                X = X.head(row_count)
            X['count'] = LabelEncoder().fit_transform(X['count'])
            y = X.pop('count')

        hyper_model = create_plain_model(with_encoder=True)
        tb = get_tool_box(X, y)
        X_train, X_test, y_train, y_test = \
            tb.train_test_split(X, y, test_size=0.3, random_state=9527)
        X_train, X_eval, y_train, y_eval = \
            tb.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

        init_kwargs = {
            'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
            **init_kwargs
        }

        compete_experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
        base_experiment = Experiment(hyper_model, X_train, y_train, **init_kwargs)

        mydict_compete = compete_experiment.get_data_character()
        mydict_base = base_experiment.get_data_character()

        assert mydict_base
        assert mydict_compete
        assert mydict_base['experimentType'] == 'base'
        assert mydict_compete['experimentType'] == 'compete'
        assert mydict_base['target']['taskType'] == 'multiclass'
        assert mydict_base['target']['freq'] is None
        assert mydict_base['target']['unique']
        assert mydict_base['target']['mean'] is None
        assert mydict_base['target']['max'] is None
        assert mydict_base['target']['min'] is None
        assert mydict_base['target']['stdev'] is None
        assert mydict_base['target']['dataType']
        assert len(mydict_base['targetDistribution']) <= 10
        assert mydict_base['datasetShape']['X_train']
        assert mydict_base['datasetShape']['y_train']
        assert mydict_base['datasetShape']['X_eval']
        assert mydict_base['datasetShape']['y_eval']
        assert mydict_base['datasetShape']['X_test']
        assert mydict_compete['featureDistribution']

    # A test for binary task
    def experiment_with_blood(self, init_kwargs, run_kwargs, row_count=3000, with_dask=False):
        if with_dask:
            X = self.blood.copy()
            y = X.pop('Class')
        else:
            X = dsutils.load_blood()
            if row_count is not None:
                X = X.head(row_count)
            X['Class'] = LabelEncoder().fit_transform(X['Class'])
            y = X.pop('Class')

        hyper_model = create_plain_model(with_encoder=True)

        tb = get_tool_box(X, y)
        X_train, X_test, y_train, y_test = \
            tb.train_test_split(X, y, test_size=0.3, random_state=9527)
        X_train, X_eval, y_train, y_eval = \
            tb.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

        init_kwargs = {
            'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
            **init_kwargs
        }

        compete_experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
        base_experiment = Experiment(hyper_model, X_train, y_train, **init_kwargs)

        mydict_compete = compete_experiment.get_data_character()
        mydict_base = base_experiment.get_data_character()

        assert mydict_base
        assert mydict_compete
        assert mydict_base['experimentType'] == 'base'
        assert mydict_compete['experimentType'] == 'compete'
        assert mydict_base['target']['taskType'] == 'binary'
        assert mydict_base['target']['freq'] is not None
        assert mydict_base['target']['unique'] == 2
        assert mydict_base['target']['mean'] is None
        assert mydict_base['target']['max'] is None
        assert mydict_base['target']['min'] is None
        assert mydict_base['target']['stdev'] is None
        assert mydict_base['target']['dataType']
        assert len(mydict_base['targetDistribution']) <= 10
        assert mydict_base['datasetShape']['X_train']
        assert mydict_base['datasetShape']['y_train']
        assert mydict_base['datasetShape']['X_eval']
        assert mydict_base['datasetShape']['y_eval']
        assert mydict_base['datasetShape']['X_test']
        assert mydict_compete['featureDistribution']

    # A test for regression task
    def experiment_with_boston(self, init_kwargs, run_kwargs, row_count=3000, with_dask=False):
        if with_dask:
            X = self.boston
            y = X.pop('target')
        else:
            X = dsutils.load_boston()
            if row_count is not None:
                X = X.head(row_count)
            X['target'] = LabelEncoder().fit_transform(X['target'])
            y = X.pop('target')
            y = y.astype('float64')

        hyper_model = create_plain_model(with_encoder=True)

        tb = get_tool_box(X, y)
        X_train, X_test, y_train, y_test = \
            tb.train_test_split(X, y, test_size=0.3, random_state=9527)
        X_train, X_eval, y_train, y_eval = \
            tb.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

        init_kwargs = {
            'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
            **init_kwargs
        }

        compete_experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
        base_experiment = Experiment(hyper_model, X_train, y_train, **init_kwargs)

        mydict_compete = compete_experiment.get_data_character()
        mydict_base = base_experiment.get_data_character()

        assert mydict_base
        assert mydict_compete
        assert mydict_base['experimentType'] == 'base'
        assert mydict_compete['experimentType'] == 'compete'
        assert mydict_base['target']['taskType'] == 'regression'
        assert mydict_base['target']['freq'] is None
        assert mydict_base['target']['unique']
        assert mydict_base['target']['mean'] is not None
        assert mydict_base['target']['max'] is not None
        assert mydict_base['target']['min'] is not None
        assert mydict_base['target']['stdev'] is not None
        assert mydict_base['target']['dataType'] == 'float'
        assert len(mydict_base['targetDistribution']) <= 10
        assert mydict_base['datasetShape']['X_train']
        assert mydict_base['datasetShape']['y_train']
        assert mydict_base['datasetShape']['X_eval']
        assert mydict_base['datasetShape']['y_eval']
        assert mydict_base['datasetShape']['X_test']
        assert mydict_compete['featureDistribution']

    def test_multiclass_with_bike_sharing(self):
        self.experiment_with_bike_sharing({}, {})

    @if_dask_ready
    def test_multiclass_with_bike_sharing_dask(self):
        self.experiment_with_bike_sharing({}, {}, with_dask=True)

    def test_binary_with_blood(self):
        self.experiment_with_blood({}, {})

    @if_dask_ready
    def test_binary_with_blood_dask(self):
        self.experiment_with_blood({}, {}, with_dask=True)

    def test_regression_with_boston(self):
        self.experiment_with_boston({}, {})

    @if_dask_ready
    def test_regression_with_boston_dask(self):
        self.experiment_with_boston({}, {}, with_dask=True)
