"""
    @Project: rock-image-transfer-learning
    @File   : rock_image_dataset_t-SNE.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
    @Date   : 2022-11-18
    @info   : rock image dimensionality reduction visualization using t-SNE
"""


from time import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from rock_image_config import RockImageConfigure
from datasetslib.CRocks import CRocks
from rock_cnn_model import RockModel

import logging


configs = RockImageConfigure()
image_size = configs.image_size
virtual_mode = "t-SNE"    # t-SNE  /  PCA
is_predict = True

is_scratch_train = configs.is_scratch_train
input_shape_vgg = (image_size, image_size, 3)


def dict_to_list(class_dict):
    class_list = []
    for key, value in class_dict.items():
        logging.info('data type of the value: {} {}'.format(type(value), value))
        if isinstance(value, (str)):
            class_list.append(value)
    return class_list


# --------------------------  Reading rock image dataset  ----------------------------
dabie = CRocks()
dabie.n_classes = configs.image_class
# resizeFolder = '%s_augu_%d' % (configs.data_folder, image_size)
resizeFolder = configs.data_folder
taglist = ['train', 'val', 'test']

x_train_files, y_train, x_val_files, y_val, x_test_files, y_test = dabie.load_auguorresizedata(force=False,
                                                                                               datafolder=resizeFolder,
                                                                                               tagList=taglist,
                                                                                               shuffle=True,
                                                                                               x_is_images=False)
total_images = len(x_train_files)
total_images_val = len(x_val_files)
total_images_test = len(x_test_files)
id2label = dabie.id2label
labels = dict_to_list(id2label)
label_num = len(labels)
print(labels)

data_files = x_train_files
target = y_train
num_rocks = total_images

data = np.array([dabie.preprocess_for_model_custom(x, image_size, image_size) for x in data_files])
print(data.shape)

checkpoints_dir = '%s_augu_%d' % (configs.checkpoints_dir, image_size)
weights_file = os.path.join(checkpoints_dir, "model.h5")

if is_predict:
    model_rock = RockModel(label_num, image_size, configs.is_scratch_train)
    input_shape = [None, image_size, image_size, 3]
    model_rock.build(input_shape=input_shape)
    model_rock.summary()
    model_rock.load_weights(weights_file)
    reconstructions = model_rock.predict(data)
    print(reconstructions.shape)
else:
    reconstructions = data.reshape((num_rocks, -1))
    print(reconstructions.shape)

t1 = time()
if virtual_mode == "t-SNE":
    data_2d = TSNE(n_components=2, learning_rate=100).fit_transform(reconstructions)
elif virtual_mode == "PCA":
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(reconstructions)

t2 = time()
t = t2 - t1
color = ['red', 'green', 'blue', 'brown', 'grey', 'orange', 'silver', 'pink', 'purple', 'grey',
         'black', 'coral', 'gold', 'wheat', 'dimgray', 'lime', 'seagreen', 'mediumseagreen', 'springgreen', 'yellow',
         'blue', 'darkblue', 'slateblue', 'navy', 'c', 'cyan', 'lightblue', 'skyblue', 'dodgerblue', 'royalblue',
         'violet', 'plum']

for i in range(label_num):
    if i > 32:
        continue
    xxx1 = data_2d[target == i, 0]
    xxx2 = data_2d[target == i, 1]
    # s：to represent the size of the marking
    plt.scatter(xxx1, xxx2, c=color[i], s=10)
plt.xlim(np.min(data_2d)-5, np.max(data_2d)+5)
plt.xlim(np.min(data_2d)-5, np.max(data_2d)+5)

plt.title('%s: %ss' % (virtual_mode, str(round(t, 2))))
# Creating scatter plot legend
plt.legend(labels=labels, title="classes", loc="lower right", fontsize=6)
svg_name = '%s_predict_tsne.svg' % configs.data_folder
plt.savefig(svg_name)
plt.show()

