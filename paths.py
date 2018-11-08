# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:08:54 2018

Initialization of paths for data, logging

@author: gretatuckute
"""

import os

def path_init():
    data_path = data_path_init()
    log_path = log_path_init()

    return data_path, log_path


def base_dir_init():
    this_base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return this_base_dir


def data_path_init():
    data_path = base_dir_init() + '\closed_loop\data'
    return data_path

def stable_path_init():
    stable_path = base_dir_init() + '\closed_loop\data\stable'
    return stable_path


def log_path_init():
    log_path = base_dir_init() + '\closed_loop\logging'
    return log_path


if __name__ == '__main__':
    base_dir = base_dir_init()
    print('====== Current dir ======')
    print(base_dir)
