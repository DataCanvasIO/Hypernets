from hypernets.conf import configure, Configurable, List, Bool, Int


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

    experiment_default_target_set = \
        List(['y', 'target'],
             allow_none=True, config=True,
             help='Default target name list.'
             )
    experiment_auto_down_sample_enabled = \
        Bool(True,
             allow_none=True, config=True,
             help=''
             )
    experiment_auto_down_sample_rows_threshold = \
        Int(10000,
            allow_none=True, config=True,
            help=''
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
