import numpy as np


def pareto_dominate(x1, x2, directions=None):
    """dominance in pareto scene, if x1 dominate x2 return True.
    """
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)

    if not isinstance(x2, np.ndarray):
        x2 = np.array(x2)

    if directions is None:
        directions = ['min'] * x1.shape[0]

    ret = []
    for i in range(x1.shape[0]):
        if directions[i] == 'min':
            if x1[i] < x2[i]:
                ret.append(1)
            elif x1[i] == x2[i]:
                ret.append(0)
            else:
                return False
        else:
            if x1[i] > x2[i]:
                ret.append(1)
            elif x1[i] == x2[i]:
                ret.append(0)
            else:
                return False

    return np.sum(np.array(ret)) >= 1


def calc_nondominated_set(solutions: np.ndarray, dominate_func=None, directions=None):

    assert solutions.ndim == 2

    if directions is None:
        directions = ['min'] * solutions.shape[1]

    if dominate_func is None:
        dominate_func = pareto_dominate

    def is_pareto_optimal(scores_i):
        if (scores_i == None).any():  # illegal individual for the None scores
            return False
        for scores_j in solutions:
            if (scores_i == scores_j).all():
                continue
            if dominate_func(x1=scores_j, x2=scores_i, directions=directions):
                return False
        return True

    optimal = []
    for i, solution in enumerate(solutions):
        if is_pareto_optimal(solution):
            optimal.append(i)
    return optimal
