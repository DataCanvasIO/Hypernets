# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import OptimizeDirection
import datetime
import os
import pickle
import shutil


class Trail():
    def __init__(self, space_sample, trail_no, reward, elapsed):
        self.space_sample = space_sample
        self.trail_no = trail_no
        self.reward = reward
        self.elapsed = elapsed


class TrailHistory():
    def __init__(self, optimize_direction):
        self.history = []
        self.optimize_direction = optimize_direction

    def append(self, trail):
        old_best = self.get_best()
        self.history.append(trail)
        new_best = self.get_best()
        improved = old_best != new_best
        return improved

    def is_existed(self, space_sample):
        return space_sample.vectors in [t.space_sample.vectors for t in self.history]

    def get_trail(self, space_sample):
        all_vectors = [t.space_sample.vectors for t in self.history]
        index = all_vectors.index(space_sample.vectors)
        if index >= 0:
            return self.history[index]
        else:
            return None

    def get_best(self):
        top1 = self.get_top(1)
        if len(top1) <= 0:
            return None
        else:
            return top1[0]

    def get_top(self, n=10):
        if len(self.history) <= 0:
            return []
        sorted_trials = sorted(self.history,
                               key=lambda
                                   t: t.reward if self.optimize_direction == OptimizeDirection.Minimize else -t.reward)
        if n > len(sorted_trials):
            n = len(sorted_trials)
        return sorted_trials[:n]

    def get_space_signatures(self):
        signatures = set()
        for s in [t.space_sample for t in self.history]:
            signatures.add(s.signature)
        return signatures

    def diff(self, trails):
        signatures = set()
        for s in [t.space_sample for t in trails]:
            signatures.add(s.signature)

        diffs = {}
        for sign in signatures:
            ts = [t for t in trails if t.space_sample.signature == sign]
            pv_dict = {}
            for t in ts:
                for p in t.space_sample.get_assigned_params():
                    k = p.alias
                    v = str(p.value)
                    if pv_dict.get(k) is None:
                        pv_dict[k] = {}
                    if pv_dict[k].get(v) is None:
                        pv_dict[k][v] = [t.reward]
                    else:
                        pv_dict[k][v].append(t.reward)
            diffs[sign] = pv_dict

        return diffs


class TrailStore(object):
    def __init__(self):
        self.reset()
        self.load()

    def put(self, dataset_id, trail):
        self.put_to_cache(dataset_id, trail)
        self._put(dataset_id, trail)

    def get_from_cache(self, dataset_id, space_sample):
        if self._cache.get(dataset_id) is None:
            self._cache[dataset_id] = {}
        dataset = self._cache[dataset_id]
        if dataset.get(space_sample.signature) is None:
            dataset[space_sample.signature] = {}
        trail = dataset[space_sample.signature].get(self.sample2key(space_sample))
        return trail

    def put_to_cache(self, dataset_id, trail):
        if self._cache.get(dataset_id) is None:
            self._cache[dataset_id] = {}
        dataset = self._cache[dataset_id]
        if dataset.get(trail.space_sample.signature) is None:
            dataset[trail.space_sample.signature] = {}
        dataset[trail.space_sample.signature][self.sample2key(trail.space_sample)] = trail

    def get(self, dataset_id, space_sample):
        trail = self.get_from_cache(dataset_id, space_sample)
        if trail is None:
            trail = self._get(dataset_id, space_sample)
            if trail is not None:
                self.put_to_cache(dataset_id, trail)
        return trail

    def _get(self, dataset_id, space_sample):
        raise NotImplementedError

    def _put(self, dataset_id, trail):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def get_all(self, dataset_id, space_signature):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def persist(self):
        raise NotImplementedError

    def sample2key(self, space_sample):
        key = ','.join([str(f) for f in space_sample.vectors])
        return key

    def check_trial(self, trail):
        pass


class DiskTrailStore(TrailStore):
    def __init__(self, home_dir=None):
        self.home_dir = self.prepare_home_dir(home_dir)
        TrailStore.__init__(self)

    def prepare_home_dir(self, home_dir):
        if home_dir is None:
            home_dir = 'trail_store'
        if home_dir[-1] == '/':
            home_dir = home_dir[:-1]
        home_dir = os.path.expanduser(home_dir)
        if not os.path.exists(home_dir):
            os.makedirs(home_dir)
        return home_dir

    def _prepare_output_dir(self, log_dir, searcher):
        if log_dir is None:
            log_dir = 'log'
        if log_dir[-1] == '/':
            log_dir = log_dir[:-1]

        running_dir = f'exp_{searcher.__class__.__name__}_{datetime.datetime.now().__format__("%m%d-%H%M%S")}'
        output_path = os.path.expanduser(f'{log_dir}/{running_dir}/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def load(self):
        pass

    def clear_history(self):
        shutil.rmtree(self.home_dir)
        self.prepare_home_dir(self.home_dir)
        self.reset()

    def reset(self):
        self._cache = {}

    def _get(self, dataset_id, space_sample):
        path = self.get_trail_path(dataset_id, space_sample)
        trail = self._load_trail(path)
        if trail is not None:
            trail.space_sample = space_sample
        return trail

    def _load_trail(self, path):
        if not os.path.exists(path):
            return None
        else:
            with open(path, 'rb') as f:
                trail = pickle.load(f)
                self.check_trial(trail)
            return trail

    def _put(self, dataset_id, trail):
        path = self.get_trail_path(dataset_id, trail.space_sample)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            temp = Trail(space_sample=None, trail_no=trail.trail_no, reward=trail.reward, elapsed=trail.elapsed)
            temp.space_sample_vectors = trail.space_sample.vectors
            pickle.dump(temp, f)

    def persist(self):
        pass

    def get_trail_path(self, dataset_id, space_sample):
        path = f'{self.home_dir}/{dataset_id}/{space_sample.signature}/{self.sample2key(space_sample)}.pkl'
        return path

    def get_all(self, dataset_id, space_signature):
        path = f'{self.home_dir}/{dataset_id}/{space_signature}'
        trails = []
        if not os.path.exists(path):
            return trails
        files = os.listdir(path)
        for f in files:
            if f.endswith('.pkl'):
                f = path + '/' + f
                trail = self._load_trail(f)
                trails.append(trail)
        return trails


default_trail_store = None


def get_default_trail_store():
    return default_trail_store


def set_default_trail_store(trail_store):
    global default_trail_store
    default_trail_store = trail_store
