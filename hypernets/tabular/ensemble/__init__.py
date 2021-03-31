# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from .base_ensemble import BaseEnsemble
from .stacking import StackingEnsemble
from .voting import AveragingEnsemble, GreedyEnsemble
from .dask_ensemble import DaskGreedyEnsemble