from hypernets.core import HyperSpace, get_random_state
from hypernets.core.searcher import OptimizeDirection, Searcher


class Individual:

    def __init__(self, dna, scores, random_state):
        self.dna = dna
        self.random_state = random_state
        self.scores = scores


class Recombination:

    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = get_random_state()
        else:
            self.random_state = random_state

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        raise NotImplementedError

    def __call__(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        # Crossover hyperparams only if they have same params
        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        assert len(params_1) == len(params_2)
        for p1, p2 in zip(params_1, params_2):
            assert p1.alias == p2.alias

        out = self.do(ind1, ind2, out_space)
        assert out.all_assigned
        return out


class SinglePointCrossOver(Recombination):

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):

        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()
        n_params = len(params_1)
        cut_i = self.random_state.randint(1, n_params - 2)  # ensure offspring has dna from both parents

        for i, hp in enumerate(out_space.params_iterator):
            if i < cut_i:
                # comes from the first parent
                hp.assign(params_1[i].value)
            else:
                hp.assign(params_2[i].value)

        return out_space


class ShuffleCrossOver(Recombination):

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        n_params = len(params_1)

        # rearrange dna & single point crossover
        cs_point = self.random_state.randint(1, n_params - 2)
        R = self.random_state.permutation(len(params_1))
        t1_params = []
        t2_params = []
        for i in range(n_params):
            if i < cs_point:
                t1_params[i] = params_1[R[i]]
                t2_params[i] = params_2[R[i]]
            else:
                t1_params[i] = params_2[R[i]]
                t2_params[i] = params_1[R[i]]

        c1_params = []
        c2_params = []
        for i in range(n_params):
            c1_params[R[i]] = c1_params[i]
            c2_params[R[i]] = c2_params[i]

        # select the first child
        for i, hp in enumerate(out_space.params_iterator):
            hp.assign(c1_params[i])

        return out_space


class UniformCrossover(Recombination):
    def __init__(self, random_state=None):
        super().__init__(random_state)
        self.p = 0.5

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):

        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        # crossover
        for i, hp in enumerate(out_space.params_iterator):
            if self.random_state.random() >= self.p:
                hp.assign(params_1[i].value)
            else:
                hp.assign(params_2[i].value)

        assert out_space.all_assigned
        return out_space


class SinglePointMutation:

    def __init__(self, random_state, proba=0.7):
        self.random_state = random_state
        self.proba = proba

    def do(self, sample_space, out_space, proba=None):

        if proba is None:
            proba = self.proba

        if self.random_state.rand(0, 1) < proba:
            return sample_space

        assert sample_space.all_assigned

        parent_params = sample_space.get_assigned_params()
        pos = self.random_state.randint(0, len(parent_params))

        # perform mutate
        for i, hp in enumerate(out_space.params_iterator):
            if i > (len(parent_params) - 1) or not parent_params[i].same_config(hp):
                hp.random_sample()
            else:
                if i == pos:
                    new_value = hp.random_sample(assign=False)
                    while new_value == parent_params[i].value:
                        new_value = hp.random_sample(assign=False)
                    hp.assign(new_value)
                else:
                    hp.assign(parent_params[i].value)

        return out_space
