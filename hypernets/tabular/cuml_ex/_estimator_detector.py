from ..estimator_detector import EstimatorDetector


class CumlEstimatorDetector(EstimatorDetector):
    def __call__(self, *args, **kwargs):
        from .. import CumlToolBox
        result = super(CumlEstimatorDetector, self).__call__(*args, **kwargs)

        estimator = self.create_estimator(self.get_estimator_cls())
        X, y = self.prepare_data()
        X, y = CumlToolBox.from_local(X, y)

        try:
            self.fit_estimator(estimator, X, y)
            result.add('fitted_with_cuml')
        except:
            pass

        return result
