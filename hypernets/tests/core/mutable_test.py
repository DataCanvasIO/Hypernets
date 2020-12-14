# -*- coding:utf-8 -*-
"""

"""

from hypernets.core.ops import Identity
from hypernets.core.search_space import *


class Test_Mutable:
    def test_scope(self):
        with HyperSpace().as_default():
            id1 = Identity()
            id2 = Identity(name='named_id')
            id3 = Identity()
            id4 = Identity(name='named_id_2')

            assert id1.name == 'Module_Identity_1'
            assert id1.id == 'Module_Identity_1'

            assert id2.name == 'named_id'
            assert id2.id == 'ID_named_id'

            assert id3.name == 'Module_Identity_2'
            assert id3.id == 'Module_Identity_2'

            assert id4.name == 'named_id_2'
            assert id4.id == 'ID_named_id_2'

            hp1 = Int(0, 100)
            hp2 = Real(0, 10.0)
            hp3 = Choice([1, 2, 3, 4])

            assert hp1.name == 'Param_Int_1'
            assert hp1.id == 'Param_Int_1'

            assert hp2.name == 'Param_Real_1'
            assert hp2.id == 'Param_Real_1'

            assert hp3.name == 'Param_Choice_1'
            assert hp3.id == 'Param_Choice_1'
