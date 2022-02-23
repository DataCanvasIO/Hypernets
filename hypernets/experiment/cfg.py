from hypernets.conf import configure, Configurable, Bool, Int, String, List, Dict


@configure()
class ExperimentCfg(Configurable):
    experiment_callbacks_console = \
        List(default_value=[],
             allow_none=True, config=True,
             help='ExperimentCallback instance or name list.'
             )
    experiment_callbacks_notebook = \
        List(default_value=['hypernets.experiment.SimpleNotebookCallback', ],
             allow_none=True, config=True,
             help='ExperimentCallback instance or name list.'
             )

    experiment_default_target_set = \
        List(default_value=['y', 'target', 'class'],
             allow_none=True, config=True,
             help='Default target names.'
             )
    experiment_auto_down_sample_enabled = \
        Bool(False,
             allow_none=True, config=True,
             help=''
             )
    experiment_auto_down_sample_rows_threshold = \
        Int(10000,
            allow_none=True, config=True,
            help=''
            )
    experiment_discriminator = \
        String('once_percentile',
               allow_none=True, config=True,
               help='discriminator identity, "percentile" or "progressive"',
               )
    experiment_discriminator_options = \
        Dict(default_value={'percentile': 50, 'min_trials': 5, 'min_steps': 5, 'stride': 1},
             key_trait=String(),
             allow_none=True, config=True,
             help='discriminator settings',
             )
    experiment_data_adaption_min_cols_limit = \
        Int(10,
            allow_none=True, config=True,
            help=''
            )

    hyper_model_callbacks_console = \
        List(default_value=['hypernets.core.callbacks.SummaryCallback', ],
             allow_none=True, config=True,
             help='Callback instance or name list.'
             )
    hyper_model_callbacks_notebook = \
        List(default_value=['hypernets.core.callbacks.NotebookCallback',
                            'hypernets.core.callbacks.ProgressiveCallback', ],
             allow_none=True, config=True,
             help='Callback instance or name list.'
             )
