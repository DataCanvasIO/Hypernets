import json

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from hypernets.utils import get_tree_importances


def test_get_tree_importances():
    X, y = load_iris(return_X_y=True)
    rfc = DecisionTreeClassifier().fit(X, y)
    print(rfc)
    imps_dict = get_tree_importances(rfc)
    assert len(imps_dict.keys()) == 4
    for c in ['col_1', 'col_2', 'col_3', 'col_0']:
        assert c in imps_dict.keys()

    values_type = list(set(map(lambda v: type(v), imps_dict.values())))

    assert len(values_type) == 1
    assert values_type[0] == int or values_type[0] == float  # not numpy type
    assert json.dumps(imps_dict)  # has only python base type
