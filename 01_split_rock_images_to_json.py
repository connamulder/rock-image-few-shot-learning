"""
    @Project: rock-image-few-shot-learning
    @File   : main_learning.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
    @info   : A specified number (num_choice_select) of rock images per class are randomly selected and
              saved as a json format file (splitfile_rock_train_2.json).
"""


import os
import json
import numpy as np


# ----------- To save the information of the rock image training dataset as json format file. ------------------
meta_savedir = 'F:/Pytorch_learning/rock-image-few-shot-learning/data'
data_rootdir = 'E:/datasets/rocks3.0.6_tvt'
num_choice_select = 36
data_tag = 'train'
data = {}
file_path = "%s.json" % data_tag
file_json_path = "splitfile_rock_%s_2.json" % data_tag

ilabels = os.listdir(os.path.join(data_rootdir, data_tag))
label2id = dict(zip(ilabels, range(len(ilabels))))
id2label = dict(zip(label2id.values(), label2id.keys()))
print(id2label)

label_names = id2label.values()
data['label_names'] = list(label_names)

image_labels = []
image_names = []
for i in range(len(id2label)):
    label = id2label[i]
    ifolder = os.path.join(data_rootdir, data_tag, label)
    print(os.listdir(ifolder))
    files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
    for f in files:
        f_name = "%s/%s/%s" % (data_tag, label, f)
        image_names.append(f_name)
        image_labels.append(i)
data['image_labels'] = image_labels
data['image_names'] = image_names

file_path = os.path.join(meta_savedir, file_path)
with open(file_path, 'w') as f_json:
    json.dump(data, f_json)


meta_path = file_path
with open(meta_path, 'r') as f:
    meta_data = json.load(f)
    print(len(meta_data['label_names']))
    print(meta_data['label_names'][:2])
    print(len(meta_data['image_labels']))
    print(meta_data['image_labels'][:2])
    print(len(meta_data['image_names']))
    print(meta_data['image_names'][:2])
    val_label = meta_data['image_labels']
    val = list(set(val_label))
    val.sort(key=val_label.index)
    print(val)

    data = []
    is_success = True
    for index in range(len(meta_data['label_names'])):
        # if index > 2:
        #     break
        idx = index
        class_ids = np.where(np.in1d(meta_data['image_labels'], index))[0]
        if len(class_ids) < num_choice_select:
            print("The class %d number < randomly selected number %d, program termination!" % (index, num_choice_select))
            is_success = False
            break
        class_ids = np.random.choice(class_ids, num_choice_select, replace=False)
        val_names = meta_data['image_names']
        class_names = np.array(val_names)[class_ids]
        # print(class_names)
        arr = class_ids.tolist()
        data.append(arr)

    if is_success:
        file_json_path = os.path.join(meta_savedir, file_json_path)
        with open(file_json_path, 'w') as f_json:
            json.dump(data, f_json)