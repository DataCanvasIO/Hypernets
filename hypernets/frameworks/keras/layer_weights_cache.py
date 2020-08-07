# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import gc


class LayerWeightsCache():
    def __init__(self):
        self.reset()
        super(LayerWeightsCache, self).__init__()

    def reset(self):
        self.cache = dict()
        self.hit_counter = 0
        self.miss_counter = 0

    def clear(self):
        del self.cache
        gc.collect()
        self.reset()

    def hit(self):
        self.hit_counter += 1

    def miss(self):
        self.miss_counter += 1

    def put(self, key, layer):
        assert self.cache.get(key) is None, f'Duplicate keys are not allowed. key:{key}'
        self.cache[key] = layer

    def retrieve(self, key):
        item = self.cache.get(key)
        if item is None:
            self.miss()
        else:
            self.hit()
        return item
