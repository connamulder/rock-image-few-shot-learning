import json
import numpy as np

NUM_VALID = 31


def process_adjacent_matrix(label_idx_file, testsetup, adjacent_matrix_file, coefficient, process_type, kg_ratio, use_all_base=False):
    '''preprocess the adjacent matrix'''
    # keep novel and base category
    with open(label_idx_file, 'r') as f:
        lowshotmeta = json.load(f)
    novel_classes = lowshotmeta['novel_classes_1']
    novel2_classes = lowshotmeta['novel_classes_2']
    base_classes = lowshotmeta['base_classes_1']
    base2_classes = lowshotmeta['base_classes_2']

    if testsetup:
        if use_all_base:
            ignore_ind = novel_classes 
            valid_nodes = novel2_classes + base2_classes + base_classes
        else:
            ignore_ind = novel_classes + base_classes
            valid_nodes = novel2_classes + base2_classes
    else:
        if use_all_base:
            ignore_ind = novel2_classes
            valid_nodes = novel_classes + base2_classes + base_classes
        else:
            ignore_ind = novel2_classes + base2_classes
            valid_nodes = novel_classes + base_classes

    mat = np.load(adjacent_matrix_file)
    num_classes = mat.shape[0]
    if process_type == 'semantic':
        in_matrix = mat

    elif process_type == 'random':
        mat = np.random.rand(num_classes, num_classes) * 0.2
        mat = np.random.rand(num_classes, num_classes) * 10
        mat[ignore_ind, :] = 0
        mat[:, ignore_ind] = 0
        in_matrix = mat
    return in_matrix


