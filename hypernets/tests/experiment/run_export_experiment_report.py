import time

from sklearn.model_selection import train_test_split

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment
from hypernets.tabular.datasets import dsutils


def main():
    df = dsutils.load_boston()

    df_train, df_eval = train_test_split(df, test_size=0.2)
    search_space = PlainSearchSpace(enable_lr=False, enable_nn=False, enable_dt=False, enable_dtr=True)

    experiment = make_experiment(PlainModel, df_train,
                                 target='target',
                                 search_space=search_space,
                                 log_level='info',
                                 random_state=8086,
                                 report_render='excel')
    estimator = experiment.run(max_trials=10)
    print(estimator)


if __name__ == '__main__':
    t = time.time()
    main()
    print(time.time() - t)
