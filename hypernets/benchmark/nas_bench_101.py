# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from ..core.ops import InputChoice, ModuleChoice, HyperInput, Identity, MultipleChoice, Choice
from ..core.search_space import HyperSpace
from ..core.meta_learner import MetaLearner
from ..core.trial import get_default_trail_store, TrailHistory, DiskTrailStore, Trail

import numpy as np
from nasbench import api


class NasBench101():
    def __init__(self, num_nodes, ops=None, input='input',
                 output='output', nasbench_filepath=None, trail_store_path=None):

        self.num_nodes = num_nodes
        if ops is None:
            ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
        self.ops = ops
        self.num_ops = len(ops)
        self.input = input
        self.output = output
        self.nasbench = api.NASBench(nasbench_filepath)
        self.trail_store_path = trail_store_path

    def get_ops(self, node_no):
        ops = [Identity(name=f'node{node_no}_ops{i}') for i in range(self.num_ops)]
        return ops

    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            input0 = HyperInput()
            inputs = [input0]
            last = input0
            for node_no in range(1, self.num_nodes):
                input_i = Identity(hp_inputs=MultipleChoice(list(range(len(inputs))),
                                                            num_chosen_most=len(inputs),
                                                            num_chosen_least=0))(last)
                if node_no < self.num_nodes - 1:
                    node_i = ModuleChoice(self.get_ops(node_no))
                    node_i(input_i)
                else:
                    node_i = Identity(name=f'output')(input_i)
                last = node_i
                inputs.append(last)
            space.set_inputs(input0)
        return space

    def sample2spec(self, space_sample):
        assert space_sample.all_assigned
        edges_choice = []
        ops_choice = []
        for hp in space_sample.assigned_params_stack:
            if isinstance(hp, MultipleChoice):  #
                edges_choice.append(hp.value)
            elif isinstance(hp, Choice):
                ops_choice.append(hp.value)
            else:
                raise ValueError(f'Unsupported hyper parameter:{hp}')

        assert len(edges_choice) == self.num_nodes - 1
        assert len(ops_choice) == self.num_nodes - 2

        matrix = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=int)
        col = 1
        for rows in edges_choice:
            for row in rows:
                matrix[row][col] = 1
            col += 1

        ops = []
        ops.append(self.input)
        for op in ops_choice:
            ops.append(self.ops[op])
        ops.append(self.output)

        return matrix, ops

    def valid_space_sample(self, space_sample):
        matrix, ops = self.sample2spec(space_sample)
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        return self.nasbench.is_valid(model_spec)

    def run_searcher(self, searcher, max_trails=None, max_time_budget=5e6, use_meta_learner=True):
        history = TrailHistory('max')
        if use_meta_learner:
            disk_trail_store = DiskTrailStore(self.trail_store_path)
            disk_trail_store.clear_history()
            meta_learner = MetaLearner(history, 'nas_bench_101', disk_trail_store)
            searcher.set_meta_learner(meta_learner)

        self.nasbench.reset_budget_counters()
        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        trail_no = 0
        while True:
            trail_no += 1
            if max_trails is not None and trail_no > max_trails:
                break

            sample = searcher.sample()
            matrix, ops = self.sample2spec(sample)
            model_spec = api.ModelSpec(matrix=matrix, ops=ops)
            data = self.nasbench.query(model_spec)

            if data['validation_accuracy'] > best_valids[-1]:
                best_valids.append(data['validation_accuracy'])
                best_tests.append(data['test_accuracy'])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])
            time_spent, _ = self.nasbench.get_budget_counters()
            times.append(time_spent)
            reward = data['test_accuracy']
            trail = Trail(sample, trail_no, reward, data['training_time'])
            history.append(trail)
            searcher.update_result(sample, reward)

            if time_spent > max_time_budget:
                # Break the first time we exceed the budget.
                break

        return times, best_valids, best_tests
