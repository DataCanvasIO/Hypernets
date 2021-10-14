# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import sys as sys_

is_os_windows = sys_.platform.find('win') == 0

from ._doc_lens import DocLens
from ._fsutils import filesystem as fs
from ._tic_tok import tic_toc, report as tic_toc_report, report_as_dataframe as tic_toc_report_as_dataframe
from .common import generate_id, combinations, isnotebook, Counter, to_repr, get_params
from .common import load_data, load_module
from ._estimators import load_estimator, save_estimator
