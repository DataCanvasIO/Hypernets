import copy
import json
import os.path
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
import xlsxwriter

from hypernets.experiment._extractor import ConfusionMatrixMeta
from hypernets.experiment.compete import DataCleanStep,DriftDetectStep, FeatureImportanceSelectionStep,\
    FeatureGenerationStep, MulticollinearityDetectStep, PermutationImportanceSelectionStep
from hypernets.utils import logging, const
from hypernets.experiment import ExperimentMeta, DatasetMeta, StepMeta, StepType
from hypernets.utils.common import human_data_size

logger = logging.get_logger(__name__)


class Theme:

    def __init__(self, theme_name):  # May support custom theme in the next version
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


FeatureTrans = namedtuple('FeatureTrans', ('feature', 'method', 'stage', 'reason', 'remark'))


class FeatureTransCollector:

    METHOD_DROP = "drop"
    METHOD_ADD = "add"

    def __init__(self, steps: List[StepMeta]):
        self.steps = steps

    @staticmethod
    def _collect_feature_trans_of_data_clean_step(step: StepMeta):
        fts = []
        if step is not None:
            for feature, reason in step.extension['unselected_reason'].items():
                fts.append(FeatureTrans(feature=feature, method=FeatureTransCollector.METHOD_DROP,
                                        stage=step.type, reason=reason, remark=None))
        return fts

    @staticmethod
    def _collect_feature_trans_of_drift_detect_step(step: StepMeta):
        fts = []
        unselected_features_drift = step.extension['unselected_features']
        over_variable_threshold_features = unselected_features_drift['over_variable_threshold']
        if over_variable_threshold_features is not None:
            for col, score in over_variable_threshold_features:
                remark = {'score': score}
                fts.append(FeatureTrans(feature=col, method='drop',
                                        stage=step.type,
                                        reason='over_variable_threshold', remark=json.dumps(remark)))
        over_threshold_features = unselected_features_drift['over_threshold']
        if over_threshold_features is not None:
            for epoch in over_threshold_features:
                for col, imp in epoch['removed_features']:
                    remark = {
                        'imp': imp,
                        'epoch': epoch['epoch'],
                        'elapsed': epoch['elapsed']
                    }
                    fts.append(FeatureTrans(feature=col, method=FeatureTransCollector.METHOD_DROP,
                                            stage=step.type,
                                            reason='over_threshold', remark=json.dumps(remark)))
        return fts

    @staticmethod
    def _collect_feature_trans_of_multicollinearity_detect_step(step: StepMeta):
        reason = 'multicollinearity_feature'
        unselected_features = step.extension['unselected_features']
        fts = []
        for k, v in unselected_features.item():
            fts.append(FeatureTrans(feature=k, method=FeatureTransCollector.METHOD_DROP,
                                    stage=step.type,
                                    reason=reason, remark=json.dumps(v)))
        return fts

    @staticmethod
    def _collect_feature_trans_of_feature_importance_selection_step(step: StepMeta):
        reason = 'low_importance'
        importances = step.extension['importances']
        fts = []
        for item in importances:
            if item['dropped']:
                _item = copy.deepcopy(item)
                del _item['name']
                fts.append(FeatureTrans(feature=item['name'], method=FeatureTransCollector.METHOD_DROP,
                                        stage=step.type,
                                        reason=reason, remark=json.dumps(_item)))
        return fts

    @staticmethod
    def _collect_feature_trans_of_feature_generation_step(step: StepMeta):
        reason = 'generated'
        importances = step.extension['outputFeatures']
        fts = []
        for item in importances:
            _item = copy.deepcopy(item)
            del _item['name']
            fts.append(FeatureTrans(feature=item['name'], method=FeatureTransCollector.METHOD_ADD,
                                    stage=step.type,
                                    reason=reason, remark=json.dumps(_item)))
        return fts

    def _get_handler(self, step):
        return self._collect_feature_trans_of_data_clean_step

    def get_handler(self, step_class_name):
        _mapping = {
            DataCleanStep: self._collect_feature_trans_of_data_clean_step,
            DriftDetectStep: self._collect_feature_trans_of_drift_detect_step,
            FeatureImportanceSelectionStep: self._collect_feature_trans_of_feature_importance_selection_step,
            FeatureGenerationStep: self._collect_feature_trans_of_feature_generation_step,
            MulticollinearityDetectStep: self._collect_feature_trans_of_multicollinearity_detect_step,
            PermutationImportanceSelectionStep: self._collect_feature_trans_of_feature_importance_selection_step,
        }

        _name_handler_mapping = {k.__name__: v for k, v in _mapping.items()}

        return _name_handler_mapping.get(step_class_name)

    def collect(self):
        fts = []
        for step in self.steps:
            handler = self.get_handler(step.type)
            if handler is not None:
                logger.debug(f"Collect feature transformation of step {step.name}")
                fts.extend(handler(step))
        return fts


class ReportRender:
    def __init__(self, **kwargs):
        pass

    def render(self, experiment_meta: ExperimentMeta, **kwargs):
        pass


class ExcelReportRender(ReportRender):
    MAX_CELL_LENGTH = 50

    def __init__(self, file_path: str = './report.xlsx', theme='default'):
        """
        Parameters
        ----------
        file_path: str, optional
            The excel report file path, default is './report.xlsx', if exists will be overwritten
        """
        super(ExcelReportRender, self).__init__(file_path=file_path)

        if os.path.exists(file_path):
            if not os.path.isfile(file_path):
                raise ValueError(f"Report excel file path already exists, and not a file: {file_path}")
            logger.warning(f"Report excel file path is already exists, it will be overwritten: {file_path}")
            os.remove(file_path)
        else:
            excel_file_dir = os.path.dirname(file_path)
            if not os.path.exists(excel_file_dir):
                logger.info(f"create directory '{excel_file_dir}' because of not exists ")
                os.makedirs(excel_file_dir, exist_ok=True)

        self.theme = Theme(theme)
        self.workbook = xlsxwriter.Workbook(os.path.abspath(file_path))  # {workbook}: {experiment_report} = 1:1

    def _write_cell(self, sheet, row_index, column_index, value, max_length_dict, cell_format_dict=None):

        if cell_format_dict is not None:
            cell_format = self.workbook.add_format(cell_format_dict)
        else:
            cell_format = None
        sheet.write(row_index, column_index, value, cell_format)

        value_width = len(str(value))
        max_len = max_length_dict.get(column_index, 0)
        if max_len < value_width:
            max_length_dict[column_index] = value_width
            if value_width > self.MAX_CELL_LENGTH:
                value_width_ = self.MAX_CELL_LENGTH
            else:
                value_width_ = value_width
            sheet.set_column(column_index, column_index, value_width_+2)

    def _render_2d_table(self, df, table_config, sheet, start_position=(0, 0)):
        """ Render pd.DataFrame to excel table

        Parameters
        ----------
        df: pd.DataFrame
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

        def calc_cell_length(header, max_content_length):
            header_len = len(header) + 4  # 2 space around header
            if max_content_length > header_len:
                if max_content_length <= self.MAX_CELL_LENGTH:
                    return max_content_length
                else:
                    return self.MAX_CELL_LENGTH
            else:
                return header_len

        max_len_dict = {}

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
                self._write_cell(sheet, cell_row_i, start_position[1],
                                 formatted_index_value, max_len_dict, index_style)

        # render header
        if write_header:
            df_len = df.applymap(lambda v: len(str(v)))
            max_len_dict = df_len.max().to_dict()
            header_render = header_config['render']
            for i, column_config in enumerate(table_config['columns']):
                formatted_column_value, column_style = header_render(i, column_config['name'], df.columns)
                if write_index:
                    y_index = start_position[1] + 1 + i
                else:
                    y_index = start_position[1] + i

                self._write_cell(sheet, start_position[0], y_index,
                                 formatted_column_value, max_len_dict, column_style)
                #  set header width automatically
                # cell_len = calc_cell_length(formatted_column_value, max_len_dict[column_config['key']])
                # sheet.set_column(i, i, cell_len)

        # render corner
        if write_corner:
            corner_render = corner_config['render']
            corner_value, corner_style = corner_render(start_position)
            self._write_cell(sheet, start_position[0], start_position[0], corner_value,
                             max_len_dict, corner_style)

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

                self._write_cell(sheet, cell_row_i, cell_col_i, formatted_value, max_len_dict, style)
        return sheet

    def _default_header_render_config(self):
        return {
            'render': lambda index, value, row: (value, self.theme.get_header_style())
        }

    @staticmethod
    def _default_cell_render(index, value, row):
        return value, {}

    @staticmethod
    def _get_keys_in_table_config(table_config):
        return [c['key'] for c in table_config['columns']]

    @staticmethod
    def _data_list_to_df(data_list: List, columns_name: List):
        def _get_value(obj, name):
            if hasattr(obj, name):
                return getattr(obj, name)
            else:
                logger.warning(f"Obj {obj} has no filed '{name}'")
                return None

        df = pd.DataFrame(data=[[_get_value(item, c) for c in columns_name]
                                for item in data_list], columns=columns_name)
        return df

    def _write_dataset_sheet(self, dataset_metas: List[DatasetMeta]):
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
                {'name': "Kind", 'key': 'kind', 'render': dataset_default_render},
                {'name': "Task", 'key': 'task', 'render': dataset_default_render},
                {
                    'name': "Shape",
                    'key': 'shape',
                    'render': lambda index, value, entity: (f"({','.join(map(lambda v:str(v), value))})", get_dataset_style(index))
                }, {
                    'name': "Memory",
                    'key': 'memory',
                    'render': lambda index, value, entity: (human_data_size(value), get_dataset_style(index))
                }
                # }, {
                #     'name': "Size",
                #     'key': 'size',
                #     'render': lambda index, value, entity: (human_data_size(value), get_dataset_style(index))
                # },
                # {'name': "File type", 'key': 'file_type', 'render': dataset_default_render},
                # {'name': "Has label", 'key': 'has_label', 'render': dataset_default_render},
                # {'name': "Remark",   'key': 'remark', 'render': dataset_default_render}
            ],
            "header": self._default_header_render_config()
        }
        columns_name = self._get_keys_in_table_config(table_config)
        df = self._data_list_to_df(dataset_metas, columns_name)
        self._render_2d_table(df, table_config, sheet_name)

    def _write_feature_transformation(self, steps: List[StepMeta]):
        fts = FeatureTransCollector(steps).collect()

        sheet_name = "Features"
        bg_colors = self.theme.theme_config['feature_trans']['bg_colors']

        def default_render(index, value, entity):
            # importance, drifted, data_clean
            style = {'bg_color': bg_colors.get(entity.stage, bg_colors['default']),
                     'align': 'center', 'border': 1, 'border_color': '#c4c7cf'}
            return value, style

        def remark_render(index, value, entity):
            if value is None:
                value = ""
            return default_render(index, value, entity)

        table_config = {
            "columns": [
                {'name': "Feature", 'key': 'feature', 'render': default_render},
                {'name': "Method", 'key': 'method', 'render': default_render},
                {'name': "Stage ", 'key': 'stage', 'render': default_render},
                {'name': "Reason ", 'key': 'reason', 'render': default_render},
                {'name': "Remark", 'key': 'remark', 'render': remark_render}
            ],
            "header": self._default_header_render_config()
        }
        columns_name = self._get_keys_in_table_config(table_config)
        # df_feature_trans = pd.DataFrame(data=[ft._asdict().values() for ft in fts], columns=fts[0]._asdict().keys())
        df = self._data_list_to_df(fts, columns_name)

        self._render_2d_table(df, table_config, sheet_name)

    def _write_confusion_matrix(self, confusion_matrix_data: ConfusionMatrixMeta):
        df = pd.DataFrame(data=confusion_matrix_data.data)

        df.columns = [str(c) for c in confusion_matrix_data.labels]
        df.index = df.columns

        sheet_name = "Confusion matrix"
        confusion_matrix_style = self.theme.theme_config['confusion_matrix']

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

    def _write_resource_usage(self, samples):
        """ Recommend sampled every min or 30 seconds
        Parameters
        ----------
        samples: ['datetime', 'cpu', 'ram'] => [(2020-10-10 22:22:22, 0.1, 0.2,)]

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
        df = pd.DataFrame(samples, columns=['datetime', 'cpu', 'ram'])

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

    def _write_ensemble(self, step: StepMeta):
        """
        Parameters
        ----------
        step:
            {
                'weight': 0.2,
                'lift': 0.2,
                'models': [ [('Age', 0.3)] ]
            }
        Returns
        -------

        """
        sheet_name = step.name

        estimators = step.extension['estimators']
        if estimators is None or len(estimators) < 1:
            logger.warning(f"Empty estimators, skipped create '{sheet_name}' sheet.")
            return

        sheet = self.workbook.add_worksheet(sheet_name)

        # flat importances to a df
        flat_df_imps = []
        for estimator in estimators:
            _model_index = estimator['index']
            _weight = estimator['weight']
            _lift = estimator['lift']
            for _cv_fold, imps in enumerate(estimator['models']):
                df_imps = pd.DataFrame(data=imps.items(), columns=['feature', 'importances'])
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
            model_end_position = model_start_position + sum(len(m) for m in models) - 1
            sheet.merge_range(model_start_position, 0, model_end_position, 0, i, cell_format=merged_cf)  # write index
            sheet.merge_range(model_start_position, 1, model_end_position, 1, estimator['weight'], cell_format=merged_cf)
            sheet.merge_range(model_start_position, 2, model_end_position, 2, estimator['lift'], cell_format=merged_cf)
            model_start_position = model_end_position + 1
            cv_enable = len(estimator['models']) > 1
            for _cv_fold, df_imps in enumerate(estimator['models']):
                # cv_start_position
                cv_end_position = len(df_imps) + cv_start_position - 1
                sheet.merge_range(cv_start_position, 3, cv_end_position, 3, _cv_fold if cv_enable else '-',
                                  cell_format=merged_cf)  # write index
                cv_start_position = cv_end_position + 1

        first_imp_df = estimators[0]['models'][0]  # first df

        # write importance chart
        chart = self.workbook.add_chart({'type': 'bar'})

        chart.add_series({
            # Note: if sheet name contains space should use quotes
            'values':  f"='{sheet_name}'!$F$2:$F${len(first_imp_df)+1}",  # importance
            "categories": f"='{sheet_name}'!$E$2:$E${len(first_imp_df)+1}",  # feature
            "name": "Feature importances of first model",
        })

        chart.set_legend({'position': 'none'})
        chart.set_y_axis({'name': 'Feature'})
        chart.set_x_axis({'name': 'Importance'})

        sheet.insert_chart('G1', chart)

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

        sheet.insert_chart('O1', chart_model)

    def _write_pseudo_labeling(self, step_meta):
        label_stats: dict = step_meta.extension['samples']
        sheet_name = "pseudo_labeling"
        labels = list(label_stats.keys())
        samples = [label_stats[l] for l in labels ]
        df = pd.DataFrame(data={'label': labels, 'samples': samples})
        table_config = {
            "columns": [
                {
                    'name': 'Label',
                    'key': 'label',
                    'render': self._default_cell_render
                }, {
                    'name': 'Samples',
                    'key': 'samples',
                    'render': self._default_cell_render
                }
            ],
            "header": self._default_header_render_config()
        }
        self._render_2d_table(df, table_config, sheet_name)

    def _write_prediction_stats(self, datasets: List[DatasetMeta], prediction_elapsed):
        sheet_name = "Prediction stats"

        # check predict elapsed
        predict_elapsed = None
        if prediction_elapsed is None or prediction_elapsed[0] <= 0:  # fix ZeroDivisionError: float division by zero
            self.log_skip_sheet(sheet_name)
            return
        else:
            predict_elapsed = prediction_elapsed[0]

        # check eval rows
        eval_data_rows = None
        if datasets is not None:
            for dataset in datasets:
                if DatasetMeta.TYPE_EVAL == dataset.kind:
                    eval_data_rows = dataset.shape[0]

        if eval_data_rows is None or eval_data_rows <= 0:
            self.log_skip_sheet(sheet_name)
            return

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
                    'render': lambda index, value, row: (round(value/row['elapsed']/1000, 2), {})
                }
            ],
            "header": {
                'render': lambda index, value, row: (value, self.theme.get_header_style())
            }
        }

        df = pd.DataFrame(data=[[DatasetMeta.TYPE_EVAL, predict_elapsed, eval_data_rows]],
                          columns=['dataset', 'elapsed', 'rows'])
        self._render_2d_table(df, table_config, sheet_name, start_position=(0, 0))

    def _write_classification_evaluation(self, report_dict, sheet_name):
        metrics_keys = ['precision', 'recall', 'f1-score', 'support']
        scores_list = []
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):  # value of 'accuracy' is float
                row = [metrics[k] for k in metrics_keys]
                row.insert(0, label)
                scores_list.append(row)
        eval_keys_with_label = metrics_keys.copy()
        eval_keys_with_label.insert(0, '')
        df = pd.DataFrame(data=scores_list, columns=eval_keys_with_label)

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

    def _write_regression_evaluation(self, evaluation_metrics, sheet_name):
        df = pd.DataFrame(data=evaluation_metrics.items(), columns=['metric', 'score'])

        table_config = {
            "columns": [
                {
                    'name': 'Metric',
                    'key': 'metric',
                    'render': self._default_cell_render
                }, {
                    'name': 'Score',
                    'key': 'score',
                    'render': self._default_cell_render
                }
            ],
            "header": self._default_header_render_config()
        }
        self._render_2d_table(df, table_config, sheet_name)

    @staticmethod
    def _get_step_by_type(exp, step_cls):
        for s in exp.steps:
            if isinstance(s, step_cls):
                return s
        return None

    @staticmethod
    def log_skip_sheet(name):
        logger.info(f"Skip create {name} sheet because of empty data. ")

    def _write_evaluation(self, task, classification_report, evaluation_metrics):
        #  experiment_meta.classification_report
        sheet_name = "Evaluation"
        if task in [const.TASK_BINARY, const.TASK_MULTICLASS]:
            if classification_report is not None:
                self._write_classification_evaluation(classification_report, sheet_name)
            else:
                self.log_skip_sheet(sheet_name)
        elif task in [const.TASK_REGRESSION]:
            if evaluation_metrics is not None:
                self._write_regression_evaluation(evaluation_metrics, sheet_name)
        else:
            logger.warning(f'Unknown task type {task}, skip to create sheet "{sheet_name}" ')

    def render(self, experiment_meta: ExperimentMeta, **kwargs):
        """Render report data into a excel file

        Parameters
        ----------
        experiment_meta:
            if part of {experiment_report} is empty maybe skip to create sheet.
        kwargs

        Returns
        -------

        """

        if experiment_meta.datasets is not None:
            self._write_dataset_sheet(experiment_meta.datasets)
        else:
            self.log_skip_sheet('datasets')

        self._write_feature_transformation(experiment_meta.steps)

        # write evaluation
        self._write_evaluation(experiment_meta.task,
                               experiment_meta.classification_report, experiment_meta.evaluation_metrics)

        if experiment_meta.confusion_matrix is not None:  # Regression has no CM
            self._write_confusion_matrix(experiment_meta.confusion_matrix)
        else:
            self.log_skip_sheet('confusion_matrix')

        self._write_prediction_stats(experiment_meta.datasets, experiment_meta.prediction_elapsed)

        if experiment_meta.resource_usage is not None:
            self._write_resource_usage(experiment_meta.resource_usage)
        else:
            self.log_skip_sheet('resource_usage')

        # write sheet by step
        for step in experiment_meta.steps:
            if step.type == StepType.Ensemble:
                self._write_ensemble(step)
            elif step.type == StepType.PseudoLabeling:
                self._write_pseudo_labeling(step)

        logger.info(f"write report excel to {self.workbook.filename}")
        self.workbook.close()

    def __getstate__(self):
        states = dict(self.__dict__)
        if 'workbook' in states:
            del states['workbook']
        return states


def get_render(name):  # instance factory
    if name == 'excel':
        return ExcelReportRender
    raise ValueError(f"Unknown render '{name}' .")

