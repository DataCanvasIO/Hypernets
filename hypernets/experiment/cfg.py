from hypernets.conf import configure, Configurable, List


@configure()
class ExperimentCfg(Configurable):
    experiment_callbacks_console = \
        List([],
             allow_none=True, config=True,
             help='ExperimentCallback instance or name list.'
             )
    experiment_callbacks_notebook = \
        List(['hypernets.experiment.SimpleNotebookCallback', ],
             allow_none=True, config=True,
             help='ExperimentCallback instance or name list.'
             )

    hyper_model_callbacks_console = \
        List(['hypernets.core.callbacks.SummaryCallback', ],
             allow_none=True, config=True,
             help='Callback instance or name list.'
             )
    hyper_model_callbacks_notebook = \
        List(['hypernets.core.callbacks.NotebookCallback', ],
             allow_none=True, config=True,
             help='Callback instance or name list.'
             )
