import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from hn_widget.experiment_util import extract_proba_density


def experiment_with_iris():
	data = load_iris()
	X = data['data']
	y = data['target']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
	knn = KNeighborsClassifier(n_neighbors = 3)
	knn.fit(X, y)
	y_proba_on_test = knn.predict_proba(X_test)
	classes = knn.classes_

	epd = extract_proba_density(0)
	probability_density = epd.get_proba_density_estimation(y_proba_on_test, classes)

	assert probability_density

def test_get_prob_density_with_iris():
	experiment_with_iris()

test_get_prob_density_with_iris()