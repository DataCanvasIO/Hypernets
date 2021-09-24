# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from ._base import get_previous_trials_scores, get_percentile_score, UnPromisingTrial, BaseDiscriminator
from .percentile import PercentileDiscriminator, ProgressivePercentileDiscriminator, OncePercentileDiscriminator

_discriminators = {
    'percentile': PercentileDiscriminator,
    'once_percentile': OncePercentileDiscriminator,
    'percentile_discriminator': PercentileDiscriminator,
    'progressive': ProgressivePercentileDiscriminator,
    'progressive_percentile': ProgressivePercentileDiscriminator,
    'progressive_percentile_discriminator': ProgressivePercentileDiscriminator,
}


def _get_discriminator_cls(identifier):
    if isinstance(identifier, str):
        cls = _discriminators.get(identifier.lower(), None)
        if cls is not None:
            return cls
    elif isinstance(identifier, type) and issubclass(identifier, BaseDiscriminator):
        return identifier

    raise ValueError(f'Illegal discriminator:{identifier}')


def make_discriminator(cls, optimize_direction='min', **kwargs):
    cls = _get_discriminator_cls(cls)

    if cls == PercentileDiscriminator:
        default_kwargs = dict(percentile=0)
    elif cls == ProgressivePercentileDiscriminator:
        default_kwargs = dict(percentile_list=[0])
    else:
        default_kwargs = {}

    kwargs = {**default_kwargs, **kwargs}
    discriminator = cls(optimize_direction=optimize_direction, **kwargs)
    return discriminator
