# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from deeptables.models.config import ModelConfig
from deeptables.models.deeptable import DeepTable
from deeptables.utils import consts as DT_consts
from hypernets.core.search_space import *
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel


class DTModuleSpace(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.space.DT_Module = self
        self.config = None

    def _build(self):
        self.config = ModelConfig(**self.param_values)
        self.is_built = True

    def _compile(self, inputs):
        return inputs

    def _on_params_ready(self):
        self._build()


class DTEstimator(Estimator):
    def __init__(self, space, **config_kwargs):
        self.config_kwargs = config_kwargs
        Estimator.__init__(self, space=space)

    def _build_model(self, space):
        config = space.DT_Module.config._replace(**self.config_kwargs)
        model = DeepTable(config)
        return model

    def summary(self):
        try:
            mi = self.model.get_model()
            if mi is not None:
                mi.model.summary()
        except(Exception) as ex:
            print('---------no summary-------------')
            print(ex)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        result = self.model.evaluate(X, y, **kwargs)
        return result


class HyperDT(HyperModel):
    def __init__(self, searcher, dispatcher=None, callbacks=[], max_trails=10,
                 reward_metric=None, max_model_size=0, **config_kwargs):
        self.config_kwargs = config_kwargs
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, max_trails=max_trails,
                            reward_metric=reward_metric)

    def _get_estimator(self, space_sample):
        estimator = DTEstimator(space_sample, **self.config_kwargs)
        return estimator


def default_space():
    space = HyperSpace()
    with space.as_default():
        DTModuleSpace(nets=MultipleChoice(['dnn_nets', 'linear', 'dcn_nets', 'fm_nets']),
                      auto_categorize=Bool(),
                      cat_remain_numeric=Bool(),
                      auto_discrete=Bool(),
                      apply_gbm_features=Bool(),
                      gbm_feature_type=Choice([DT_consts.GBM_FEATURE_TYPE_DENSE, DT_consts.GBM_FEATURE_TYPE_EMB]),
                      embeddings_output_dim=Choice([4, 10, 20]),
                      embedding_dropout=Choice([0, 0.25, 0.5]),
                      stacking_op=Choice([DT_consts.STACKING_OP_ADD, DT_consts.STACKING_OP_CONCAT]),
                      output_use_bias=Bool(),
                      apply_class_weight=Bool(),
                      earlystopping_patience=Choice([1, 3, 5]))
    return space

    # categorical_columns='auto',
    # exclude_columns=[],
    # pos_label=None,
    # metrics=['accuracy'],
    # auto_categorize=False,
    # cat_exponent=0.5,
    # cat_remain_numeric=True,
    # auto_encode_label=True,
    # auto_imputation=True,
    # auto_discrete=False,
    # apply_gbm_features=False,
    # gbm_params={},
    # gbm_feature_type=DT_consts.GBM_FEATURE_TYPE_EMB,  # embedding/dense
    # fixed_embedding_dim=True,
    # embeddings_output_dim=4,
    # embeddings_initializer='uniform',
    # embeddings_regularizer=None,
    # embeddings_activity_regularizer=None,
    # dense_dropout=0,
    # embedding_dropout=0.3,
    # stacking_op=DT_consts.STACKING_OP_ADD,
    # output_use_bias=True,
    # apply_class_weight=False,
    # optimizer='auto',
    # loss='auto',
    # dnn_params={
    #     'dnn_units': ((128, 0, False), (64, 0, False)),
    #     'dnn_activation': 'relu',
    # },
    # autoint_params={
    #     'num_attention': 3,
    #     'num_heads': 1,
    #     'dropout_rate': 0,
    #     'use_residual': True,
    # },
    # fgcnn_params={'fg_filters': (14, 16),
    #               'fg_heights': (7, 7),
    #               'fg_pool_heights': (2, 2),
    #               'fg_new_feat_filters': (2, 2),
    #               },
    # fibinet_params={
    #     'senet_pooling_op': 'mean',
    #     'senet_reduction_ratio': 3,
    #     'bilinear_type': 'field_interaction',
    # },
    # cross_params={
    #     'num_cross_layer': 4,
    # },
    # pnn_params={
    #     'outer_product_kernel_type': 'mat',
    # },
    # afm_params={
    #     'attention_factor': 4,
    #     'dropout_rate': 0
    # },
    # cin_params={
    #     'cross_layer_size': (128, 128),
    #     'activation': 'relu',
    #     'use_residual': False,
    #     'use_bias': False,
    #     'direct': False,
    #     'reduce_D': False,
    # },
    # home_dir=None,
    # monitor_metric=None,
    # earlystopping_patience=1,
    # gpu_usage_strategy=DT_consts.GPU_USAGE_STRATEGY_GROWTH,
    # distribute_strategy=None,
