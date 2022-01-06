import os
from collections import OrderedDict

import psutil

from hypernets.utils import get_perf


def test_get_perf():
    proc = psutil.Process(os.getpid())
    perf = get_perf(proc)
    assert isinstance(perf, OrderedDict)
    assert 'cpu_total' in perf.keys()
