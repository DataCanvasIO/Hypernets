import pandas as pd
import xlsxwriter
import copy

from hypernets.utils import logging
from hypernets.utils.common import human_data_size

logger = logging.get_logger(__name__)


class ExperimentReport:
    def __init__(self, datasets=None, feature_trans=None, evaluation_metric=None,
                 confusion_matrix=None, resource_usage=None, feature_importances=None, prediction_stats=None):
        self.datasets = datasets
        self.feature_trans = feature_trans
        self.evaluation_metric = evaluation_metric
        self.confusion_matrix = confusion_matrix
        self.resource_usage = resource_usage
        self.feature_importances = feature_importances
        self.prediction_stats = prediction_stats


class Theme:

    def __init__(self):  # May support custom theme in the next version
        self.theme_config = {
            'common': {
                'header': {
                    'bg_color': '#012060',
                    'font_color': '#f1f7fb',
                    'align': 'center',
                    'border': 1,
                    'border_color': '#c4c7cf'
                },
                'row_diff': [
                  {'bg_color': '#ffffff'},
                  {'bg_color': '#dae1f3'},
                ]
            },
            'feature_trans': {
                'bg_colors': {
                    'DataCleanStep': "#be4242",
                    'DriftDetectStep': "#ffc000",
                    'FeatureImportanceSelectionStep': '#ffff00',
                    'default': '#ffffff'
                }
            },
            'confusion_matrix': {
                'prediction': {
                    'bg_color': '#E26B0A',
                    'font_color': '#ffffff'
                },
                'actual': {
                    'bg_color': '#00B050',
                    'font_color': '#ffffff'
                },
                'data': {
                    'bg_color': '#538DD5',
                    'font_color': '#ffffff'
                }
            }
        }

    def get_header_style(self):
        return self.theme_config['common']['header']

    def get_row_diff_style(self):
        return self.theme_config['common']['row_diff']


class ExcelReportRender:

    def __init__(self, file_path):
        self.theme = Theme()
        self.workbook = xlsxwriter.Workbook(file_path)

    def _render_2d_table(self, df, table_config, sheet, start_position=(0, 0)):
        """ Render entities to excel table

        Parameters
        ----------
        df
        table_config:

            {
                "columns": [
                    {'name': "Feature name", 'key': 'type', 'render': lambda index, value, row: (display_value, style)}
                ],
                "index": {
                    'render': lambda index_of_index, index_value, index_values: (display_value, style),
                    'corner_render': lambda position: (display_value, style)
                },
                "header": {
                    'render': lambda index, value, row: (display_value, style)
                }
            }

        sheet: str instance or sheet
        start_position

        Returns
        -------

        """
        assert len(start_position) == 2, "start_position should be Tuple[int, int]"

        index_config = table_config.get('index')
        header_config = table_config.get('header')
        corner_config = table_config.get('corner')

        write_index = index_config is not None
        write_header = header_config is not None
        write_corner = write_index and write_header and corner_config is not None

        # create sheet
        if isinstance(sheet, str):
            sheet = self.workbook.add_worksheet(sheet)

        # render index
        index_config = table_config.get('index')
        if write_index:
            # write index
            index_render = index_config['render']
            for i, index_value in enumerate(df.index):
                formatted_index_value, index_style = index_render(i, index_value, df.index)
                if write_header:
                    cell_row_i = start_position[0] + 1 + i
                else:
                    cell_row_i = start_position[0] + i
                sheet.write(cell_row_i, start_position[1], formatted_index_value,
                            self.workbook.add_format(index_style))

        # render header
        if write_header:
            header_render = header_config['render']
            for i, column_config in enumerate(table_config['columns']):
                formatted_column_value, column_style = header_render(i, column_config['name'], df.columns)
                if write_index:
                    y_index = start_position[1] + 1 + i
                else:
                    y_index = start_position[1] + i
                sheet.write(start_position[0], y_index,
                            formatted_column_value, self.workbook.add_format(column_style))
                #  set header width automatically
                sheet.set_column(i, i, len(formatted_column_value) + 4)

        # render corner
        if write_corner:
            corner_render = corner_config['render']
            corner_value, corner_style = corner_render(start_position)
            sheet.write(start_position[0], start_position[0], corner_value, self.workbook.add_format(corner_style))

        for i, (series_i, series_row) in enumerate(df.iterrows()):
            for j, column in enumerate(table_config['columns']):
                value = series_row[column['key']]
                render = column['render']
                # 返回的value必须是excel可以接受的格式，比如int/flot/datetime 这些类型
                formatted_value, style = render(i, value, series_row)
                if write_header:
                    cell_row_i = i + 1 + start_position[0]
                else:
                    cell_row_i = i + start_position[0]

                if write_index:
                    cell_col_i = j + 1 + start_position[1]
                else:
                    cell_col_i = j + start_position[1]
                sheet.write(cell_row_i, cell_col_i, formatted_value, self.workbook.add_format(style))
        return sheet

    def _default_header_render_config(self):
        return {
            'render': lambda index, value, row: (value, self.theme.get_header_style())
        }

    @staticmethod
    def _default_cell_render(index, value, row):
        return value, {}

    @staticmethod
    def _check_input_df(df: pd.DataFrame, sheet_name, keys=None):
        if df.shape[0] < 1 or df.shape[1] < 1:
            logger.warning(f"Empty DataFrame, skipped create '{sheet_name}' sheet.")
            return False
        if keys is not None:
            for k in keys:
                assert k in df.columns, f'Require column "{k}" in df to create sheet "{sheet_name}"'
        return True  # check pass

    @staticmethod
    def _get_keys_in_table_config(table_config):
        return [c['key'] for c in table_config['columns']]

    def _write_dataset_sheet(self, df: pd.DataFrame):
        sheet_name = "Datasets"

        def get_dataset_style(index):
            data_row_gb_color = self.theme.get_row_diff_style()
            row_gb_color_ = data_row_gb_color[index % len(data_row_gb_color)]
            style_ = {'align': 'center', 'border': 1, 'border_color': '#c4c7cf'}
            style_.update(row_gb_color_)
            return style_

        def dataset_default_render(index, value, entity):
            return value, get_dataset_style(index)

        table_config = {
            "columns": [
                {'name': "Feature name", 'key': 'type', 'render': dataset_default_render},
                {'name': "Task", 'key': 'task', 'render': dataset_default_render},
                {
                    'name': "Shape",
                    'key': 'shape',
                    'render': lambda index, value, entity: (f"({','.join(map(lambda v:str(v), value))})", get_dataset_style(index))
                }, {
                    'name': "Memory",
                    'key': 'memory',
                    'render': lambda index, value, entity: (human_data_size(value), get_dataset_style(index))
                }, {
                    'name': "Size",
                    'key': 'size',
                    'render': lambda index, value, entity: (human_data_size(value), get_dataset_style(index))
                },
                {'name': "File type", 'key': 'file_type', 'render': dataset_default_render},
                {'name': "Has label", 'key': 'has_label', 'render': dataset_default_render},
                {'name': "Remark",   'key': 'remark', 'render': dataset_default_render}
            ],
            "header": self._default_header_render_config()
        }
        if not self._check_input_df(df, sheet_name, self._get_keys_in_table_config(table_config)):
            return

        self._render_2d_table(df, table_config, sheet_name)

    def _write_feature_transformation(self, df: pd.DataFrame):
        sheet_name = "Features"
        bg_colors = self.theme.theme_config['feature_trans']['bg_colors']

        def default_render(index, value, entity):
            # importance, drifted, data_clean
            style = {'bg_color': bg_colors.get(entity.stage, bg_colors['default']),
                     'align': 'center', 'border': 1, 'border_color': '#c4c7cf'}
            return value, style

        table_config = {
            "columns": [
                {'name': "Feature", 'key': 'name', 'render': default_render},
                {'name': "Column", 'key': 'col', 'render': default_render},
                {'name': "Method", 'key': 'method', 'render': default_render},
                {'name': "Stage ", 'key': 'stage', 'render': default_render},
                {'name': "Reason ", 'key': 'reason', 'render': default_render},
                {'name': "Remark", 'key': 'remark', 'render': default_render}
            ],
            "header": self._default_header_render_config()
        }

        if not self._check_input_df(df, sheet_name, keys=self._get_keys_in_table_config(table_config)):
            return

        self._render_2d_table(df, table_config, sheet_name)

    def _write_confusion_matrix(self, df: pd.DataFrame):
        sheet_name = "Confusion matrix"
        confusion_matrix_style = self.theme.theme_config['confusion_matrix']
        if not self._check_input_df(df, sheet_name):
            return

        def to_config(c):
            return {
                'name': c,
                'key': c,
                'render': lambda index, value, entity: (value, confusion_matrix_style['data'])
            }
        header_style = copy.deepcopy(confusion_matrix_style['actual'])
        header_style['align'] = 'right'
        table_config = {
            "columns": [to_config(c) for c in df.columns],
            "index": {
                'render': lambda ii, value, i_values: (value, confusion_matrix_style['prediction']),
            },
            "header": {
                'render': lambda index, value, row: (value, header_style)
            },
            'corner': {
                'render': lambda position: ("", confusion_matrix_style['data'])
            }
        }
        sheet = self._render_2d_table(df, table_config, sheet_name, start_position=(0, 0))

        # write legends
        legend_row_start = len(df.columns) + 1 + 2
        legend_col = len(df.columns)
        sheet.write(legend_row_start+0, legend_col, "Legends", self.workbook.add_format({'bold': True}))
        sheet.write(legend_row_start+1, legend_col, "Actual",
                    self.workbook.add_format(confusion_matrix_style['actual']))
        sheet.write(legend_row_start+2, legend_col, "Predict",
                    self.workbook.add_format(confusion_matrix_style['prediction']))

    def _write_resource_usage(self, df: pd.DataFrame):
        """ Recommend sampled every min or 30 seconds
        Parameters
        ----------
        df

        Returns
        -------

        """
        sheet_name = "Resource usage"
        table_config = {
            "columns": [
                {
                    'name': 'Datetime',
                    'key': 'datetime',
                    'render': lambda index, value, row: (value.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S'), {})
                }, {
                    'name': 'CPU',
                    'key': 'cpu',
                    'render': lambda index, value, row: (value, {})
                }, {
                    'name': 'RAM',
                    'key': 'ram',
                    'render': lambda index, value, row: (value, {})
                }
            ],
            "header": {
                'render': lambda index, value, row: (value, self.theme.get_header_style())
            }
        }

        if not self._check_input_df(df, sheet_name, keys=self._get_keys_in_table_config(table_config)):
            return

        sheet = self._render_2d_table(df, table_config, sheet_name, start_position=(0, 0))

        # write chart
        chart = self.workbook.add_chart({'type': 'line'})

        chart.add_series({
            # Note: if sheet name contains space should use quotes
            'values': f"='{sheet_name}'!$B$2:$B${df.shape[0]+1}",
            "categories": f"='{sheet_name}'!$A$2:$A${df.shape[0]+1}",
            "name": "CPU",
            # "name": f"={sheet_name}!$B$1",
        })
        # Configure a primary (default) Axis.
        chart.add_series({
            "values": f"='{sheet_name}'!$C$2:$C${df.shape[0]+1}",
            "categories": f"='{sheet_name}'!$A$2:$A${df.shape[0] + 1}",
            "y2_axis": True,  # Configure a series with a secondary axis.
            "name": "RAM"
        })

        chart.set_legend({'position': 'top'})

        chart.set_y_axis({'name': 'CPU(%)'})
        chart.set_y2_axis({'name': 'RAM(MB)'})
        # Handle dates example
        # dd-mm-yy hh:mm:ss
        # chart.set_x_axis({'name': 'Datetime', 'date_axis': True, 'num_format': 'dd-mm-yy hh:mm:ss'})
        chart.set_x_axis({'name': 'Datetime'})

        sheet.insert_chart('E2', chart)

        return sheet

    def _write_feature_importance(self, estimators):
        """
        Parameters
        ----------
        estimators:
            {
                'weight': 0.2,
                'lift': 0.2,
                'models': [{imps_df}]
            }
            imps_df:
                col_name, feature, importances
                age     , 年龄    , 0.3
        Returns
        -------

        """
        sheet_name = "Feature importance"
        required_imps_keys = ['feature', 'col_name', 'importances']

        if estimators is None or len(estimators) < 1:
            logger.warning(f"Empty estimators, skipped create '{sheet_name}' sheet.")
            return

        def _check_imps_df(df, model_index, cv_fold):
            for c in required_imps_keys:
                assert c in df.columns, f'For model_index={model_index} cv_fold={cv_fold}, ' \
                                                f'column {c} not exists .'
        sheet = self.workbook.add_worksheet(sheet_name)

        # flat importances to a df
        flat_df_imps = []
        for _model_index, estimator in enumerate(estimators):
            _weight = estimator['weight']
            _lift = estimator['lift']
            for _cv_fold, df_imps in enumerate(estimator['models']):
                _check_imps_df(df_imps, _model_index, _cv_fold)
                df_imps['model_index'] = [_model_index for k in range(df_imps.shape[0])]
                df_imps['weight'] = [_weight for k in range(df_imps.shape[0])]
                df_imps['lift'] = [_lift for k in range(df_imps.shape[0])]
                df_imps['cv_fold'] = [_cv_fold for k in range(df_imps.shape[0])]
                flat_df_imps.append(df_imps)
        df_flatted = pd.concat(flat_df_imps, axis=0)

        table_config = {
            "columns": [
                {
                    'name': 'Model index',
                    'key': 'model_index',
                    'render': self._default_cell_render
                }, {
                    'name': 'Weight',
                    'key': 'weight',
                    'render': self._default_cell_render
                }, {
                    'name': 'Lift',
                    'key': 'lift',
                    'render': self._default_cell_render
                }, {
                    'name': 'CV fold',
                    'key': 'cv_fold',
                    'render': self._default_cell_render
                }, {
                    'name': 'Column',
                    'key': 'col_name',
                    'render': self._default_cell_render
                }, {
                    'name': 'Feature',
                    'key': 'feature',
                    'render': self._default_cell_render
                },  {
                    'name': 'Importances',
                    'key': 'importances',
                    'render': self._default_cell_render
                }
            ],
            "header": self._default_header_render_config()
        }
        sheet = self._render_2d_table(df_flatted, table_config, sheet)

        # merge 'model_index', 'weight', 'lift', 'cv_fold' cells
        model_start_position = 1  # first row is header
        cv_start_position = 1
        for i, estimator in enumerate(estimators):
            models = estimator['models']
            merged_cf = self.workbook.add_format({'align': 'center', 'valign': 'vcenter', })
            model_end_position = model_start_position + sum(m.shape[0] for m in models) - 1
            sheet.merge_range(model_start_position, 0, model_end_position, 0, i, cell_format=merged_cf)  # write index
            sheet.merge_range(model_start_position, 1, model_end_position, 1, estimator['weight'], cell_format=merged_cf)
            sheet.merge_range(model_start_position, 2, model_end_position, 2, estimator['lift'], cell_format=merged_cf)
            model_start_position = model_end_position + 1
            for _cv_fold, df_imps in enumerate(estimator['models']):
                # cv_start_position
                cv_end_position = df_imps.shape[0] + cv_start_position - 1
                sheet.merge_range(cv_start_position, 3, cv_end_position, 3, _cv_fold,
                                  cell_format=merged_cf)  # write index
                cv_start_position = cv_end_position + 1

        first_imp_df = estimators[0]['models'][0]  # first df

        # write importance chart
        chart = self.workbook.add_chart({'type': 'bar'})

        chart.add_series({
            # Note: if sheet name contains space should use quotes
            'values':  f"='{sheet_name}'!$G$2:$G${first_imp_df.shape[0]+1}",  # importance
            "categories": f"='{sheet_name}'!$F$2:$F${first_imp_df.shape[0]+1}",  # feature
            "name": "Feature importances of first model",
        })

        chart.set_legend({'position': 'none'})
        chart.set_y_axis({'name': 'Feature'})
        chart.set_x_axis({'name': 'Importance'})

        sheet.insert_chart('H1', chart)

        # write weights chart
        chart_model = self.workbook.add_chart({'type': 'column'})
        chart_model.set_title({'name': 'Weights & Lift'})
        chart_model.add_series({
            # Note: if sheet name contains space should use quotes
            'values': f"='{sheet_name}'!$B$2:$B${df_flatted.shape[0] + 1}",  # weight
            "categories": f"='{sheet_name}'!$A$2:$A${df_flatted.shape[0] + 1}",  # model index
            "name": "Weights",
        })

        chart_model.add_series({
            'values': f"='{sheet_name}'!$C$2:$C${df_flatted.shape[0] + 1}",  # lift
            "categories": f"='{sheet_name}'!$A$2:$A${df_flatted.shape[0] + 1}",  # model index
            "name": "Lift",
        })

        chart_model.set_legend({'position': 'left'})
        # chart_weight.set_y_axis({'name': 'Weight'})
        # chart_weight.set_y2_axis({'name': 'Lift'})
        chart_model.set_x_axis({'name': 'Model index'})

        sheet.insert_chart('P1', chart_model)

    def _write_prediction_stats(self, df):
        """
        Parameters
        ----------
        df:
            dataset, elapsed,   rows
               Test,    1000, 100000
        Returns
        -------
        """
        sheet_name = "Prediction stats"

        table_config = {
            "columns": [
                {
                    'name': 'Dataset',
                    'key': 'dataset',
                    'render': lambda index, value, row: (value, {})
                }, {
                    'name': 'Elapsed seconds',
                    'key': 'elapsed',
                    'render': lambda index, value, row: (value, {})
                }, {
                    'name': 'Rows',
                    'key': 'rows',
                    'render': lambda index, value, row: (value, {})
                }, {
                    'name': 'Speed(K/s)',
                    'key': 'rows',
                    'render': lambda index, value, row: (round(value/row['elapsed']/1024, 2), {})
                }
            ],
            "header": {
                'render': lambda index, value, row: (value, self.theme.get_header_style())
            }
        }
        if not self._check_input_df(df, sheet_name, keys=self._get_keys_in_table_config(table_config)):
            return

        self._render_2d_table(df, table_config, sheet_name, start_position=(0, 0))

    def _write_evaluation_metric(self, df: pd.DataFrame):
        sheet_name = "Evaluation"
        if not self._check_input_df(df, sheet_name):
            return

        def _to_config(c):
            return {
                'name': c,
                'key': c,
                'render': self._default_cell_render
            }
        table_config = {
            "columns": [_to_config(c) for c in df.columns],
            "header": self._default_header_render_config()
        }
        self._render_2d_table(df, table_config, sheet_name)

    def render(self, experiment_report: ExperimentReport):
        """Render report data into a excel file.

        Parameters
        ----------
        experiment_report:
            if data(usually is a pd.DataFrame) is empty will skip to create sheet.

        Returns
        -------
        """
        if experiment_report.datasets is not None:
            self._write_dataset_sheet(experiment_report.datasets)

        if experiment_report.feature_trans is not None:
            self._write_feature_transformation(experiment_report.feature_trans)

        if experiment_report.evaluation_metric is not None:
            self._write_evaluation_metric(experiment_report.evaluation_metric)

        if experiment_report.confusion_matrix is not None:  # Regression has no CM
            self._write_confusion_matrix(experiment_report.confusion_matrix)

        if experiment_report.resource_usage is not None:
            self._write_resource_usage(experiment_report.resource_usage)

        if experiment_report.feature_importances is not None:
            self._write_feature_importance(experiment_report.feature_importances)

        if experiment_report.prediction_stats is not None:
            self._write_prediction_stats(experiment_report.prediction_stats)

        self.workbook.close()
