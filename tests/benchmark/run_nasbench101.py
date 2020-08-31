# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from nasbench import api
from hypernets.benchmark.nas_bench_101 import NasBench101
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.evolution_searcher import EvolutionSearcher
from hypernets.core.meta_learner import MetaLearner
from hypernets.core.trial import get_default_trail_store, TrailHistory, DiskTrailStore, Trail
from tests import test_output_dir

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Path to the file nasbench.tfrecord')
args = parser.parse_args()

nasbench = api.NASBench(args.input_file)
hyn_nasbench = NasBench101(7, ops=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])


def valid_space_sample(space_sample):
    matrix, ops = hyn_nasbench.sample2spec(space_sample)
    model_spec = api.ModelSpec(matrix=matrix, ops=ops)
    return nasbench.is_valid(model_spec)


def run_searcher(searcher, max_trails=None, max_time_budget=5e6, use_meta_learner=True):
    history = TrailHistory('max')
    if use_meta_learner:
        disk_trail_store = DiskTrailStore(f'{test_output_dir}/trail_store')
        disk_trail_store.clear_history()
        meta_learner = MetaLearner(history, 'nas_bench_101', disk_trail_store)
        searcher.set_meta_learner(meta_learner)

    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    trail_no = 0
    while True:
        trail_no += 1
        if max_trails is not None and trail_no > max_trails:
            break

        sample = searcher.sample()
        matrix, ops = hyn_nasbench.sample2spec(sample)
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        data = nasbench.query(model_spec)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        reward = data['test_accuracy']
        trail = Trail(sample, trail_no, reward, data['training_time'])
        history.append(trail)
        searcher.update_result(sample, reward)

        if time_spent > max_time_budget:
            # Break the first time we exceed the budget.
            break

    return times, best_valids, best_tests


searcher = MCTSSearcher(hyn_nasbench.get_space, optimize_direction='max', space_sample_validation_fn=valid_space_sample)

# searcher = RandomSearcher(hyn_nasbench.get_space, space_sample_validation_fn=valid_space_sample)

times, best_v, best_t = run_searcher(searcher)
print(times)
