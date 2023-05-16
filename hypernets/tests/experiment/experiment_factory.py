import numpy as np
from sklearn.metrics import get_scorer, make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.utils import const


def _create_experiment(predefined_kwargs, maker=None, need_test=False, user_kwargs=None):
    df = dsutils.load_boston().head(1000)
    df['Constant'] = [0 for i in range(df.shape[0])]
    df['Id'] = [i for i in range(df.shape[0])]
    target = 'target'
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1234)
    df_test.pop(target)
    df_train['Drifted'] = np.random.random(df_train.shape[0])
    df_test['Drifted'] = np.random.random(df_test.shape[0]) * 100

    def maker_(*args, **kwargs):
        if 'random_state' not in kwargs.keys():
            kwargs['random_state'] = 1234
        return make_experiment(PlainModel, *args, **kwargs)

    default_kwargs = dict(
        log_level='info',
    )
    predefined_kwargs.update(default_kwargs)
    if maker is None:
        maker = maker_
        predefined_kwargs['search_space'] = PlainSearchSpace(enable_lr=False,
                                                             enable_nn=False, enable_dt=False, enable_dtr=True)
    if need_test:
        predefined_kwargs['test_data'] = df_test

    predefined_kwargs.update(user_kwargs)

    return maker(df_train, target=target, task=const.TASK_REGRESSION, **predefined_kwargs)


def create_disable_cv_experiment(maker=None, **user_kwargs):
    data_cleaner_args = {'drop_duplicated_columns': True}
    exp_kwargs = dict(data_cleaner_args=data_cleaner_args,
                      cv=False,
                      drift_detection=False)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


class MyRewardMetric:
    __name__ = 'foo'

    def __hash__(self):
        return hash(self.__name__)

    def __eq__(self, other):
        return self.__name__ == str(other)

    def __call__(self, y_true, y_preds):
        return r2_score(y_true, y_preds)  # replace this line with yours


def my_reward_metric_func(y_true, y_preds):
    return r2_score(y_true, y_preds)  # replace this line with yours


def create_custom_reward_metric_class_experiment(maker=None, **user_kwargs):

    my_reward_metric = MyRewardMetric()
    my_scorer = make_scorer(my_reward_metric, greater_is_better=True, needs_proba=False)

    data_cleaner_args = {'drop_duplicated_columns': True}
    exp_kwargs = dict(data_cleaner_args=data_cleaner_args,
                      reward_metric=my_reward_metric,
                      scorer=my_scorer,
                      drift_detection=False)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


def create_custom_reward_metric_func_experiment(maker=None, **user_kwargs):

    my_scorer = make_scorer(my_reward_metric_func, greater_is_better=True, needs_proba=False)

    data_cleaner_args = {'drop_duplicated_columns': True}
    exp_kwargs = dict(data_cleaner_args=data_cleaner_args,
                      reward_metric=my_reward_metric_func,
                      scorer=my_scorer,
                      drift_detection=False)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


def create_data_clean_experiment(maker=None, **user_kwargs):
    data_cleaner_args = {'drop_duplicated_columns': True}
    exp_kwargs = dict(data_cleaner_args=data_cleaner_args,
                      drift_detection=False)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


def create_multicollinearity_detect_experiment(maker=None, **user_kwargs):
    exp_kwargs = dict(drift_detection=False,
                      collinearity_detection=True)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


def create_feature_generation_experiment(maker=None, **user_kwargs):
    exp_kwargs = \
        dict(feature_generation=True,
             feature_generation_trans_primitives=["cross_categorical", "add_numeric", "subtract_numeric"],
             # feature_generation_fix_input=False,
             feature_generation_max_depth=1,
             # feature_generation_categories_cols=['job', 'education'],
             # feature_generation_continuous_cols=['balance', 'duration'],
             feature_generation_datetime_cols=None,
             feature_generation_latlong_cols=None,
             feature_generation_text_cols=None,
             drift_detection=False,
             collinearity_detection=False)
    return _create_experiment(exp_kwargs, maker=maker, need_test=False, user_kwargs=user_kwargs)


def create_drift_detection_experiment(maker=None, **user_kwargs):
    exp_kwargs = \
        dict(drift_detection=True,
             drift_detection_variable_shift_threshold=0.6,
             drift_detection_threshold=0.5,
             drift_detection_min_features=6,
             drift_detection_remove_size=0.2,
             collinearity_detection=False)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


def create_feature_selection_experiment(maker=None, **user_kwargs):
    exp_kwargs = \
        dict(feature_selection=True,
             feature_selection_strategy='quantile',
             feature_selection_threshold=100,
             feature_selection_quantile=0.5,
             feature_selection_number=None, )
    return _create_experiment(exp_kwargs, maker=maker, need_test=False, user_kwargs=user_kwargs)


def create_feature_reselection_experiment(maker=None, **user_kwargs):
    exp_kwargs = \
        dict(scorer=get_scorer('neg_median_absolute_error'),
             drift_detection=False,
             feature_reselection=True,
             feature_reselection_estimator_size=10,
             feature_reselection_strategy='quantile',
             feature_reselection_threshold=0.1,
             feature_reselection_quantile=0.5,
             feature_reselection_number=None)
    return _create_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)


def _create_bankdata_experiment(predefined_kwargs, maker=None, need_test=False, user_kwargs=None):
    target = 'y'
    df = dsutils.load_bank().head(2000)
    df[target] = LabelEncoder().fit_transform(df[target])
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=9527)

    def maker_(*args, **kwargs):

        return make_experiment(PlainModel, *args, **kwargs)

    default_kwargs = dict(log_level='info')

    predefined_kwargs.update(default_kwargs)

    if maker is None:
        maker = maker_
        predefined_kwargs['search_space'] = PlainSearchSpace(enable_lr=True,
                                                             enable_nn=False, enable_dt=False, enable_dtr=False)
        predefined_kwargs['hyper_model_options'] = {'transformer': MultiLabelEncoder}

    if need_test:
        predefined_kwargs['test_data'] = df_test

    predefined_kwargs.update(user_kwargs)

    return maker(df_train, target=target, task=const.TASK_BINARY, **predefined_kwargs)


def create_pseudo_labeling_experiment(maker=None, **user_kwargs):
    exp_kwargs = \
        dict(pseudo_labeling=True,
             pseudo_labeling_strategy='threshold',
             pseudo_labeling_proba_threshold=0.5,
             pseudo_labeling_proba_quantile=None,
             pseudo_labeling_sample_number=None,
             pseudo_labeling_resplit=False,
             drift_detection=False)

    return _create_bankdata_experiment(exp_kwargs, maker=maker, need_test=True, user_kwargs=user_kwargs)
