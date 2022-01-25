# -*- coding:utf-8 -*-
"""

"""
import numpy as np
import pandas as pd

from . import if_dask_ready, is_dask_installed

if is_dask_installed:
    import dask.dataframe as dd


def prepare_dataframe():
    from hypernets.tabular.datasets import dsutils
    pdf = dsutils.load_bank()
    ddf = dd.from_pandas(pdf, npartitions=2)

    return pdf, ddf


@if_dask_ready
def test_max_abs_scale():
    from sklearn import preprocessing as sk_pre
    import hypernets.tabular.dask_ex as de

    TOL = 0.00001

    pdf, ddf = prepare_dataframe()

    num_columns = [k for k, t in pdf.dtypes.items()
                   if t in (np.int32, np.int64, np.float32, np.float64)]
    pdf = pdf[num_columns]
    ddf = ddf[num_columns]

    sk_s = sk_pre.MaxAbsScaler()
    sk_r = sk_s.fit_transform(pdf)

    de_s = de.MaxAbsScaler()
    de_r = de_s.fit_transform(ddf)

    delta = (sk_s.scale_ - de_s.scale_).abs().max()
    assert delta < TOL

    delta = (sk_r - de_r.compute()).abs().max().max()
    assert delta < TOL

    delta = (sk_s.inverse_transform(sk_r) - de_s.inverse_transform(de_r).compute()) \
        .abs().max().max()
    assert delta < TOL


@if_dask_ready
def test_ordinal_encoder():
    from hypernets.tabular.dask_ex import SafeOrdinalEncoder
    df1 = pd.DataFrame({"A": [1, 2, 3, 4],
                        "B": ['a', 'a', 'a', 'b']})
    df2 = pd.DataFrame({"A": [1, 2, 3, 5],
                        "B": ['a', 'b', 'z', '0']})

    ec = SafeOrdinalEncoder(dtype=np.int32)
    df = ec.fit_transform(dd.from_pandas(df1, npartitions=2)).compute()
    df_expect = pd.DataFrame({"A": [1, 2, 3, 4],
                              "B": [1, 1, 1, 2]})
    # diff = (df - df_expect).values
    # assert np.count_nonzero(diff) == 0
    assert np.where(df_expect.values == df.values, 0, 1).sum() == 0

    df = ec.transform(dd.from_pandas(df2, npartitions=1)).compute()
    df_expect = pd.DataFrame({"A": [1, 2, 3, 5],
                              "B": [1, 2, 0, 0]})
    assert np.where(df_expect.values == df.values, 0, 1).sum() == 0

    df = ec.inverse_transform(dd.from_pandas(df_expect, npartitions=1)).compute()
    df_expect = pd.DataFrame({"A": [1, 2, 3, 5],
                              "B": ['a', 'b', None, None]})
    assert np.where(df_expect.values == df.values, 0, 1).sum() == 0
