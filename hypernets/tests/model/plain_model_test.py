from hypernets.examples.plain_model import train_heart_disease


def test_train_heart_disease():
    train_heart_disease(cv=False, max_trials=5)


def test_train_heart_disease_with_cv():
    train_heart_disease(cv=True, max_trials=5)
