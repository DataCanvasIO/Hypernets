# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.discriminators import PercentileDiscriminator, get_previous_trials_scores, get_percentile_score
from hypernets.core import TrialHistory, Trial

history = TrialHistory(optimize_direction='min')
group_id = 'lightgbm_cv_1'
group_id2 = 'lightgbm_cv_2'
t1 = Trial(None, 1, 0.9, 0, succeeded=True)
t1.iteration_scores[group_id] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3]
t2 = Trial(None, 1, 0.8, 0, succeeded=True)
t2.iteration_scores[group_id] = [0.9, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.35, 0.35, 0.25]
t3 = Trial(None, 1, 0.8, 0, succeeded=True)
t3.iteration_scores[group_id] = [0.9, 0.85, 0.75, 0.65, 0.54, 0.44, 0.34, 0.34, 0.34, 0.24]
t4 = Trial(None, 1, 0.8, 0, succeeded=True)
t4.iteration_scores[group_id] = [0.9, 0.85, 0.75, 0.65, 0.53, 0.43, 0.33, 0.33, 0.33, 0.23]
t5 = Trial(None, 1, 0.8, 0, succeeded=True)
t5.iteration_scores[group_id] = [0.9, 0.85, 0.75, 0.65, 0.52, 0.42, 0.32, 0.32, 0.32, 0.22]
t6 = Trial(None, 1, 0.8, 0, succeeded=True)
t6.iteration_scores[group_id] = [0.9, 0.85, 0.75, 0.65, 0.51, 0.41, 0.31, 0.31, 0.31, 0.21]
t6.iteration_scores[group_id2] = [0.9, 0.85, 0.75, 0.65, 0.51, 0.41, 0.31, 0.31, 0.31, 0.21]
history.append(t1)
history.append(t2)
history.append(t3)
history.append(t4)
history.append(t5)
history.append(t6)