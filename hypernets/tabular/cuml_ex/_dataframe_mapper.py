# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy
import numpy as np
from sklearn.pipeline import _name_estimators

from hypernets.tabular.dataframe_mapper import DataFrameMapper, TransformerPipeline
from ._transformer import Localizable


class CumlTransformerPipeline(TransformerPipeline):
    def as_local(self):
        steps = [(name, tf.as_local()) for name, tf in self.steps]
        target = TransformerPipeline(steps)
        return target


def make_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.
    """
    return CumlTransformerPipeline(_name_estimators(steps))


class CumlDataFrameMapper(DataFrameMapper, Localizable):
    @staticmethod
    def _build_transformer(transformers):
        if isinstance(transformers, list):
            transformers = make_transformer_pipeline(*transformers)
        return transformers

    def _to_df(self, X, extracted, columns):
        dfs = [cudf.DataFrame(arr, index=None) for arr in extracted]
        for df, pos in zip(dfs, np.cumsum([d.shape[1] for d in dfs])):
            df.reset_index(drop=True, inplace=True)
            df.columns = [f'c{i}' for i in range(pos - df.shape[1], pos)]
        df_out = cudf.concat(dfs, axis=1, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if len(X) == len(df_out):
            df_out.index = X.index
        df_out.columns = columns

        return df_out

    @staticmethod
    def _hstack_array(extracted):
        arrs = [arr.values if isinstance(arr, cudf.DataFrame) else arr for arr in extracted]
        return cupy.hstack(arrs)

    @staticmethod
    def _fix_feature(fea):
        if isinstance(fea, (np.ndarray, cupy.ndarray)) and len(fea.shape) == 1:
            fea = fea.reshape(-1, 1)
        return fea

    def as_local(self):
        target = DataFrameMapper([], default=None, df_out=self.df_out, input_df=self.input_df,
                                 df_out_dtype_transforms=self.df_out_dtype_transforms)
        target.fitted_features_ = [(cols, t.as_local(), opts) for cols, t, opts in self.fitted_features_]
        return target
