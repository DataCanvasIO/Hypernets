# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.search_space import ModuleSpace, Choice
from hypernets.core.ops import ConnectionSpace

from sklearn import preprocessing as sk_pre
from sklearn import impute
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
import numpy as np


class PipelineInput(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.output_id = None


class PipelineOutput(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.input_id = None


class Pipeline(ConnectionSpace):
    def __init__(self, module_list, keep_link=False, space=None, name=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 0, f'module_list contains at least 1 Module.'
        assert all([isinstance(m, (ModuleSpace, list)) for m in
                    module_list]), 'module_list can only contains ModuleSpace or list.'
        self._module_list = module_list
        self.hp_lazy = Choice([0])
        ConnectionSpace.__init__(self, self.pipeline_fn, keep_link, space, name, hp_lazy=self.hp_lazy)

    def pipeline_fn(self, m):
        last = self._module_list[0]
        for i in range(1, len(self._module_list)):
            self.connect_module_or_subgraph(last, self._module_list[i])
            # self._module_list[i](last)
            last = self._module_list[i]
        pipeline_input = PipelineInput(name=self.id + '_input')
        pipeline_output = PipelineOutput(name=self.id + '_output')
        pipeline_input.output_id = pipeline_output.id
        pipeline_output.input_id = pipeline_input.id

        input = self.space.get_sub_graph_inputs(last)
        assert len(input) == 1, 'Pipeline does not support branching.'
        output = self.space.get_sub_graph_outputs(last)
        assert len(output) == 1, 'Pipeline does not support branching.'

        input[0](pipeline_input)
        pipeline_output(output[0])

        return pipeline_input, pipeline_output


class HyperTransformer(ModuleSpace):
    def __init__(self, transformer=None, space=None, name=None, **hyperparams):
        self.transformer = transformer
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        if self.transformer is not None:
            pv = self.param_values
            self.compile_fn = self.transformer(**pv)
        self.is_built = True

    def _compile(self, inputs):
        return self.compile_fn

    def _on_params_ready(self):
        pass
        # self._build()


class ComposeTransformer(HyperTransformer):
    def __init__(self, last_transformer, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, None, space, name, **kwargs)
        self.last_transformer = last_transformer
        self(last_transformer)

    def compose(self):
        raise NotImplementedError

    def get_transformers(self):
        transformers = []
        next = self.last_transformer
        while True:
            transformer = next
            assert isinstance(next, HyperTransformer)
            if isinstance(next, ComposeTransformer):
                transformer = next.compose()
            transformers.insert(0, transformer)
            inputs = self.space.get_inputs(next)
            if len(inputs) <= 0:
                break
            assert len(inputs) == 1, 'Pipeline does not support branching.'
            next = inputs[0]
        return transformers


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
