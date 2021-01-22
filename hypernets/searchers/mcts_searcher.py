# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .mcts_core import *
from ..core.searcher import Searcher, OptimizeDirection


class MCTSSearcher(Searcher):
    """
    References
    ----------
        Wang, Linnan, et al. "Alphax: exploring neural architectures with deep neural networks and monte carlo tree search." arXiv preprint arXiv:1903.11059 (2019).
        Browne, Cameron B., et al. "A survey of monte carlo tree search methods." IEEE Transactions on Computational Intelligence and AI in games 4.1 (2012): 1-43.
    """

    def __init__(self, space_fn, policy=None, max_node_space=10, candidates_size=10,
                 optimize_direction=OptimizeDirection.Minimize, use_meta_learner=True, space_sample_validation_fn=None):
        """
        :param space_fn: Callable
            A search space function which when called returns a `HyperSpace` object.
        :param policy: hypernets.searchers.mcts_core.BasePolicy, (default=None)
            The policy for *Selection* and *Backpropagation* phases, `UCT` by default.
        :param max_node_space: int, (default=10)
            Maximum space for node expansion
        :param candidates_size: int, (default=10)
            The number of samples for the meta-learner to evaluate candidate paths when roll out
        :param optimize_direction: 'min' or 'max', (default='min')
            Whether the search process is approaching the maximum or minimum reward value
        :param use_meta_learner: bool, (default=True)
            Meta-learner aims to evaluate the performance of unseen samples based on previously evaluated samples. It provides a practical solution to accurately estimate a search branch with many simulations without involving the actual training
        :param space_sample_validation_fn: Callable or None, (default=None)
            Used to verify the validity of samples from the search space, and can be used to add specific constraint rules to the search space to reduce the size of the space
        """
        if policy is None:
            policy = UCT()
        self.tree = MCTree(space_fn, policy, max_node_space=max_node_space)
        Searcher.__init__(self, space_fn, optimize_direction, use_meta_learner=use_meta_learner,
                          space_sample_validation_fn=space_sample_validation_fn)
        self.nodes_map = {}
        self.candidate_size = candidates_size

    def parallelizable(self):
        return self.use_meta_learner and self.meta_learner is not None

    def sample(self):
        # print('Sample')
        _, best_node = self.tree.selection_and_expansion()
        # print(f'Sample: {best_node.info()}')

        if self.use_meta_learner and self.meta_learner is not None:
            space_sample, candidate_sim_score, candidates_avg_score = self._select_best_candidate(best_node)
            # support for parallelize sampling
            self.tree.back_propagation(best_node, candidates_avg_score, is_simulation=True)
        else:
            space_sample = self._roll_out(best_node)

        self.nodes_map[space_sample.space_id] = best_node
        return space_sample

    def _roll_out(self, node):
        def sample():
            space_sample = self.tree.node_to_space(node)
            space_sample = self.tree.roll_out(space_sample, node)
            return space_sample

        space_sample = self._sample_and_check(sample_fn=sample)
        return space_sample

    def _select_best_candidate(self, node):
        candidates = []
        scores = []
        for i in range(self.candidate_size):
            candidate = self._roll_out(node)
            candidates.append(candidate)
            scores.append(self.meta_learner.predict(candidate, 0.5))
        index = np.argmax(scores)
        candidate_sim_score = scores[index]
        candidates_avg_score = np.average(scores)
        # print(f'selected candidates scores:{scores}, argmax:{index}')
        return candidates[index], candidate_sim_score, candidates_avg_score

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space_sample, result):
        best_node = self.nodes_map[space_sample.space_id]
        # print(f'Update result: space:{space_sample.space_id}, result:{result}, node:{best_node.info()}')
        self.tree.back_propagation(best_node, result)
        # print(f'After back propagation: {best_node.info()}')
        # print('\n\n')
        if self.use_meta_learner and self.meta_learner is not None:
            assert self.meta_learner is not None
            self.meta_learner.new_sample(space_sample)

    def summary(self):
        return str(self.tree.root)

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
