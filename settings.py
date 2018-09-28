# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:08:54 2018

Template for data/script paths

@author: gretatuckute
"""

import os

def path_init():
    data_path = data_path_init()
    # log_path = log_path_init()
    # experiment_data_path = experiment_data_init()
    # model_path = model_path_init()
    return data_path #, log_path, experiment_data_path, model_path


def base_dir_init():
    this_base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return this_base_dir
#
#
def data_path_init():
    data_path = base_dir_init() + '\closed_loop\data'
    return data_path
#
#
#def log_path_init():
#    log_path = data_path_init() + 'logs/'
#    return log_path
#
#
#def experiment_data_init():
#    experiment_data_path = os.path.normpath(data_path_init() + 'experiment_data')
#    return experiment_data_path
#
#
#def model_path_init():
#    model_path = os.path.normpath(base_dir_init() + '/mrcode/models')
#    return model_path


if __name__ == '__main__':
    base_dir = base_dir_init()
    print('====== Current dir ======')
    print(base_dir)
