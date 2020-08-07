# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""


def compile_layer(search_space, layer_class, name, **kwargs):
    if kwargs.get('name') is None:
        kwargs['name'] = name

    # In the weights sharing mode, the instance is first retrieved from the cache
    cache = search_space.__dict__.get('weights_cache')
    if cache is not None:
        layer = cache.retrieve(kwargs['name'])
        if layer is None:
            layer = layer_class(**kwargs)
            cache.put(kwargs['name'], layer)
    else:
        layer = layer_class(**kwargs)

    return layer
