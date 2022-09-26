# -*- coding:utf-8 -*-
"""

"""
import time
from _datetime import datetime
from collections import OrderedDict

import pandas as pd
import psutil

try:
    import pynvml

    pynvml_installed = True
except ImportError:
    pynvml_installed = False
except:
    import traceback

    traceback.print_exc()
    pynvml_installed = False

_gpu_devices = []


def _initialize_pynvml():
    global _gpu_devices, pynvml_installed

    if pynvml_installed and pynvml.nvmlLib is None:
        try:
            pynvml.nvmlInit()

            _gpu_devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as e:
            print('nvmlInit Error:', e)
            pynvml_installed = False
            _gpu_devices = []


def get_perf(proc=None, recursive=True, children_pool=None, metrics=None):
    """
    Get process performance metrics
    """
    _initialize_pynvml()

    if metrics is None:
        metrics = OrderedDict()

    metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    metrics['cpu_total'] = psutil.cpu_count()
    metrics['cpu_used'] = psutil.cpu_percent()
    if proc:
        if recursive:
            percents = []
            _recursive_proc(percents, proc, children_pool, lambda p: p.cpu_percent())
            metrics['proc_count'] = len(percents)
            metrics['cpu_used_proc'] = sum(percents)
        else:
            metrics['cpu_used_proc'] = proc.cpu_percent()

    mem = psutil.virtual_memory()
    metrics['mem_total'] = mem.total
    metrics['mem_used'] = mem.used
    if proc:
        if recursive:
            rss = []
            _recursive_proc(rss, proc, children_pool, lambda p: p.memory_info().rss)
            metrics['mem_used_proc'] = sum(rss)
        else:
            metrics['mem_used_proc'] = proc.memory_info().rss
    try:
        for i, h in enumerate(_gpu_devices):
            used = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            metrics[f'gpu_{i}_used'] = used.gpu
            metrics[f'gpu_{i}_mem_used'] = mem.used  # used.memory
            metrics[f'gpu_{i}_mem_total'] = mem.total
            metrics[f'gpu_{i}_power_used'] = pynvml.nvmlDeviceGetPowerUsage(h)
            metrics[f'gpu_{i}_power_total'] = pynvml.nvmlDeviceGetPowerManagementLimit(h)
    except:
        pass

    return metrics


def _recursive_proc(result_buf, proc_, children_pool, fn):
    assert children_pool is None or isinstance(children_pool, dict)

    try:
        result_buf.append(fn(proc_))

        for c in proc_.children(recursive=True):
            # _recursive_proc(result_buf, c, fn)
            if children_pool is None:
                p = c
            else:
                cpid = c.pid
                if cpid in children_pool.keys():
                    p = children_pool[cpid]
                else:
                    children_pool[cpid] = c
                    p = c

            try:
                result_buf.append(fn(p))
            except KeyboardInterrupt:
                raise
            except InterruptedError:
                raise
            except psutil.NoSuchProcess:
                pass
            except:
                import traceback
                traceback.print_exc()
                pass
    except KeyboardInterrupt:
        raise
    except InterruptedError:
        raise
    except psutil.NoSuchProcess:
        raise
    except:
        import traceback
        traceback.print_exc()
        pass


def dump_perf(file_path, pid=None, recursive=False, interval=1):
    """
    Dump process performance metrics data into a csv file
    """
    children_pool = {} if recursive else None
    metrics = OrderedDict()
    proc = psutil.Process(pid) if pid else None
    header = True
    try:
        with open(file_path, 'w') as f:
            while True:
                get_perf(proc, recursive=recursive, children_pool=children_pool, metrics=metrics)
                if header:
                    header = False
                    f.write(','.join(metrics.keys()) + '\n')
                f.write(','.join(map(str, metrics.values())) + '\n')
                f.flush()
                time.sleep(interval)
    except KeyboardInterrupt:
        # print('KeyboardInterrupt')
        pass
    except InterruptedError:
        # print('InterruptedError')
        pass
    except psutil.NoSuchProcess:
        # print('NoSuchProcess')
        pass
    except:
        import traceback
        traceback.print_exc()
        pass


def load_perf(file_path, human_readable=True):
    df = pd.read_csv(file_path)
    columns = df.columns.to_list()
    assert 'timestamp' in columns

    df['timestamp'] = df['timestamp'].apply(lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))
    start_at = df['timestamp'].min()
    delta = pd.Timedelta(1, 'S')
    df.insert(1, 'elapsed', (df['timestamp'] - start_at) // delta)

    if human_readable:
        GB = 1024 ** 3
        if 'cpu_used_proc' in columns:
            df['cpu_used_proc'] = df['cpu_used_proc'] / df['cpu_total']
        for c in columns:
            if c.startswith('mem_'):
                df[c] = df[c] / GB
            if c.startswith('gpu_'):
                if c.find('_power_') > 0:
                    df[c] = df[c] / 1000
                elif c.find('_mem_') > 0:
                    df[c] = df[c] / GB

    return df
