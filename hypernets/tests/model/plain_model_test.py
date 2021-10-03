from hypernets.core.callbacks import SummaryCallback
from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.examples.plain_model import train_heart_disease
from hypernets.searchers import make_searcher
from hypernets.tabular.sklearn_ex import MultiLabelEncoder


class DaskPlainModel(PlainModel):
    def _get_estimator(self, space_sample):
        from hypernets.tabular import get_tool_box
        import dask.dataframe as dd

        estimator = super()._get_estimator(space_sample)

        return get_tool_box(dd.DataFrame).wrap_local_estimator(estimator)


def create_plain_model(reward_metric='auc', optimize_direction='max',
                       with_encoder=False, with_dask=False):
    search_space = PlainSearchSpace(enable_dt=True, enable_lr=True, enable_nn=False)
    searcher = make_searcher('random', search_space_fn=search_space, optimize_direction=optimize_direction)

    encoder = MultiLabelEncoder if with_encoder else None
    cls = DaskPlainModel if with_dask else PlainModel
    hyper_model = cls(searcher=searcher, reward_metric=reward_metric, callbacks=[SummaryCallback()],
                      transformer=encoder)

    return hyper_model


def test_train_heart_disease():
    train_heart_disease(cv=False, max_trials=5)


def test_train_heart_disease_with_cv():
    train_heart_disease(cv=True, max_trials=5)
