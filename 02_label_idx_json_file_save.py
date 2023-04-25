"""
    @Project: rock-image-few-shot-learning
    @File   : main_learning.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
    @info   : Save few-shot learning data in a json format file.
"""


import json


file_path = 'data/label_idx_rocks_24-8-1.json'
label_names = ['022_Gabbro', '031_Diorite', '041_Syenite', '044_Monzonite', '061_Syenogranite',
               '062_Monzonitic_granite', '063_Granodiorite', '111_Basalt', '121_Andesite', '131_trachyte',
               '141_Rhyolite', '144_Obsidian', '145_Pitchstone', '147_Pumice', '210_Phyllite',
               '221_quartz_schist', '231_plagiogneiss', '233_k-feldspar_gneiss', '240_Granulite', '251_Quartzite',
               '252_Jadeite_quartzite', '260_Marble', '271_Amphibolite', '280_Eclogite', '287_Serpentinite',
               '291_Mylonite', '301_Conglomerate', '310_Sandstone', '330_Mudstone', '340_Shale',
               '356_micrite', '370_Silicalite']
base_classes = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
base_classes_1 = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17]
novel_classes_1 = [7, 8, 9, 10, 11, 12, 13, 18, 20, 27]
base_classes_2 = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
novel_classes_2 = [7, 8, 9, 10, 11, 12, 13, 20]

data_json = {}
data_json['label_names'] = label_names
data_json['base_classes'] = base_classes
data_json['base_classes_1'] = base_classes_1
data_json['novel_classes_1'] = novel_classes_1
data_json['base_classes_2'] = base_classes_2
data_json['novel_classes_2'] = novel_classes_2

with open(file_path, 'w') as f_json:
    json.dump(data_json, f_json)