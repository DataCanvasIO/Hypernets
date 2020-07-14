# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.search_space import ModuleSpace

from sklearn import preprocessing as sk_pre
from sklearn import impute
import numpy as np


class HyperTransformer(ModuleSpace):
    def __init__(self, transformer, space=None, name=None, **hyperparams):
        self.transformer = transformer
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        pv = self.param_values
        self.compile_fn = self.transformer(**pv)
        self.is_built = True

    def _compile(self, inputs):
        return self.compile_fn

    def _on_params_ready(self):
        pass
        # self._build()


class StandardScaler(HyperTransformer):
    def __init__(self, copy=True, with_mean=True, with_std=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if with_mean is not None and with_mean != True:
            kwargs['with_mean'] = with_mean
        if with_std is not None and with_std != True:
            kwargs['with_std'] = with_std
        HyperTransformer.__init__(self, sk_pre.StandardScaler, space, name, **kwargs)


class SimpleImputer(HyperTransformer):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True,
                 add_indicator=False, space=None, name=None, **kwargs):
        if missing_values is not None and missing_values != np.nan:
            kwargs['missing_values'] = missing_values
        if strategy is not None and strategy != "mean":
            kwargs['strategy'] = strategy
        if fill_value is not None:
            kwargs['fill_value'] = fill_value
        if verbose is not None and verbose != 0:
            kwargs['verbose'] = verbose
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if add_indicator is not None and add_indicator != False:
            kwargs['add_indicator'] = add_indicator

        HyperTransformer.__init__(self, impute.SimpleImputer, space, name, **kwargs)

# from sklearn.preprocessing import Binarizer, KBinsDiscretizer, MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler, \
#     LabelEncoder
