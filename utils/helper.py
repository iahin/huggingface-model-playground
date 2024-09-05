import os

import torch


def get_curr_dir(path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(current_dir, path)
    return relative_path

def create_dir_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def file_exist(path):
    if os.path.exists(path):
        return True
    return False
