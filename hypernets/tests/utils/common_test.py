import os.path
import os

from hypernets.utils import common as common_util


def test_camel_keys_to_snake():
    input_dict = {
        'datasetConf': {
            'trainData': './train.csv'
        },
        'name': 'with-feature-selection',
        'jobs': [
            {
                'featureSelection': {
                    'leastFeatures': 10
                },
                'callbackSetting': [{
                    'className': 'hypernets.core.ConsoleCallback'
                }]
            }
        ]
    }

    ret_dict = common_util.camel_keys_to_snake(input_dict)
    assert ret_dict['dataset_conf']['train_data'] == input_dict['datasetConf']['trainData']
    assert ret_dict['name'] == input_dict['name']

    input_job_conf_dict = input_dict['jobs'][0]
    ret_job_conf_dict = ret_dict['jobs'][0]

    assert ret_job_conf_dict['feature_selection']['least_features'] == \
           input_job_conf_dict['featureSelection']['leastFeatures']

    assert ret_job_conf_dict['callback_setting'][0]['class_name'] == \
           input_job_conf_dict['callbackSetting'][0]['className']


def test_make_tempfile():

    temp_file_path: str = common_util.get_temp_file_path(prefix='prefix', suffix='.txt')
    assert not os.path.exists(temp_file_path)

    assert os.path.basename(temp_file_path).startswith('prefix')
    assert os.path.basename(temp_file_path).endswith('.txt')

    temp_file_dir_created = common_util.get_temp_dir_path(prefix='prefix', suffix='prefix', create=True)
    assert os.path.exists(temp_file_dir_created)

    temp_file_dir_not_created = common_util.get_temp_dir_path(prefix='prefix', suffix='prefix', create=False)
    assert not os.path.exists(temp_file_dir_not_created)
