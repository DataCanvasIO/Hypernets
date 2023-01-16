from hypernets.core import HyperSpace, get_random_state
from hypernets.searchers.moead_searcher import Individual


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

        # cross over
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
    def __init__(self):
        super().__init__()
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
