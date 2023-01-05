import copy
import pickle

import numpy as np
from sklearn.pipeline import Pipeline

from . import fs


def save_estimator(estimator, model_path):
    if isinstance(estimator, Pipeline) and hasattr(estimator.steps[-1][1], 'save') \
            and hasattr(estimator.steps[-1][1], 'load'):
        if fs.exists(model_path):
            fs.rm(model_path, recursive=True)
        fs.mkdirs(model_path, exist_ok=True)
        if not model_path.endswith(fs.sep):
            model_path = model_path + fs.sep

        stub = copy.copy(estimator)
        stub.steps[-1][1].save(f'{model_path}pipeline.model')
        with fs.open(f'{model_path}pipeline.pkl', 'wb') as f:
            pickle.dump(stub, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with fs.open(model_path, 'wb') as f:
            pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_estimator(model_path):
    model_path_ = model_path
    if not model_path_.endswith(fs.sep):
        model_path_ = model_path_ + fs.sep

    if fs.exists(f'{model_path_}pipeline.pkl'):
        with fs.open(f'{model_path_}pipeline.pkl', 'rb') as f:
            stub = pickle.load(f)
            assert isinstance(stub, Pipeline)

        estimator = stub.steps[-1][1]
        if fs.exists(f'{model_path_}pipeline.model') and hasattr(estimator, 'load'):
            est = estimator.load(f'{model_path_}pipeline.model')
            steps = stub.steps[:-1] + [(stub.steps[-1][0], est)]
            stub = Pipeline(steps)
    else:
        with fs.open(model_path, 'rb') as f:
            stub = pickle.load(f)

    return stub


def get_tree_importances(tree_model):
    def catch_exception(func):
        def _wrapper(model):
            try:
                return func(model)
            except Exception as e:
                return False

        return _wrapper

    @catch_exception
    def is_light_gbm_model(m):
        from lightgbm.sklearn import LGBMModel
        return isinstance(m, LGBMModel)

    @catch_exception
    def is_xgboost_model(m):
        from xgboost.sklearn import XGBModel
        return isinstance(m, XGBModel)

    @catch_exception
    def is_catboost_model(m):
        from catboost.core import CatBoost
        return isinstance(m, CatBoost)

    @catch_exception
    def is_decision_tree_model(m):
        from sklearn.tree import BaseDecisionTree
        return isinstance(m, BaseDecisionTree)

    def is_numpy_num_type(v):
        for t in [int, float, np.int32, np.int64, np.float32, np.float64]:
            if isinstance(v, t) is True:
                return True
            else:
                continue
        return False

    def get_imp(n_features):
        try:
            return tree_model.feature_importances_
        except Exception as e:
            return [0 for i in range(n_features)]

    if is_xgboost_model(tree_model):
        importances_pairs = list(zip(tree_model._Booster.feature_names,
                                     get_imp(len(tree_model._Booster.feature_names))))
    elif is_light_gbm_model(tree_model):
        if hasattr(tree_model, 'feature_name_'):
            names = tree_model.feature_name_
        else:
            names = [f'col_{i}' for i in range(tree_model.feature_importances_.shape[0])]
        importances_pairs = list(zip(names, get_imp(len(names))))
    elif is_catboost_model(tree_model):
        importances_pairs = list(zip(tree_model.feature_names_, get_imp(len(tree_model.feature_names_))))
    elif is_decision_tree_model(tree_model):
        importances_pairs = [(f'col_{i}', tree_model.feature_importances_[i])
                             for i in range(tree_model.feature_importances_.shape[0])]
    else:
        importances_pairs = []

    importances = {}

    for name, imp in importances_pairs:
        if is_numpy_num_type(imp):
            imp_value = imp.tolist()  # int64, float32, float64 has tolist
        elif isinstance(imp, float) or isinstance(imp, int):
            imp_value = imp
        else:
            imp_value = float(imp)  # force convert to float
        importances[name] = imp_value

    return importances
