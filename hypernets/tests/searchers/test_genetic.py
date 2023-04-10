from hypernets.core import get_random_state, set_random_state, HyperSpace, Identity, Bool, Optional, Real, HyperInput, Choice, Int
from hypernets.searchers.genetic import SinglePointCrossOver, ShuffleCrossOver, UniformCrossover, Individual


class TestCrossOver:

    @classmethod
    def setup_class(cls):
        set_random_state(1234)
        cls.random_state = get_random_state()

    def test_shuffle_crossover(self):
        co = ShuffleCrossOver(random_state=self.random_state)
        self.run_crossover(co)

    def test_single_point_crossover(self):
        co = SinglePointCrossOver(random_state=self.random_state)
        self.run_crossover(co)

    def test_uniform_crossover(self):
        co = UniformCrossover(random_state=self.random_state)
        try:
            self.run_crossover(co)
            # P(off=[A or B]) = 0.5 ^ 3 * 2
        except Exception as e:
            print(e)

    def run_crossover(self, crossover):
        # 1. prepare data
        random_state = self.random_state

        # 2. construct a search space
        def get_space():
            space = HyperSpace()
            with space.as_default():
                input1 = HyperInput(name="input1")
                id1 = Identity(p1=Choice([1, 2, 3, 4]), p2=Int(1, 100), name="id1")
                id2 = Identity(p3=Real(0, 1), name="id2")
                id1(input1)
                id2(id1)
            return space
        out = get_space()
        print(out)

        # 3. construct individuals
        dna1 = get_space()
        dna1.assign_by_vectors([0, 50, 0.2])
        ind1 = Individual(dna=dna1, scores=[1, 1], random_state=random_state)

        dna2 = get_space()
        dna2.assign_by_vectors([1, 30, 0.5])
        ind2 = Individual(dna=dna2, scores=[1, 1], random_state=random_state)

        output = crossover(ind1=ind1, ind2=ind2, out_space=get_space())
        assert output.all_assigned

        # the offspring is not same as any parents
        assert output.vectors != ind1.dna.vectors
        assert output.vectors != ind2.dna.vectors

