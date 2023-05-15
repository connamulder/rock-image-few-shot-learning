"""
    @Project: rock-image-transfer-learning
    @File   : save_rock_images_features.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
    @Date   : 2022-11-18
    @info   : Extracting features from rock images and saving them as hdf5 files.
"""


import numpy as np
import tensorflow as tf
from datasetslib.CRocks import CRocks
import argparse
import time
from rock_cnn_model import RockModel_NoDense, RockModel
import os
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description='Save features')
    parser.add_argument('--data_folder', default='rocks3.0.6_tvt', help='dataset file')
    parser.add_argument('--model_file', default='data/model_VGG16_24class.h5', help='trained model file')
    parser.add_argument('--mode_tag', default='test', help='train / valid / test')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--image_class', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(123)
    print("NumPy:{}".format(np.__version__))
    print("TensorFlow:{}".format(tf.__version__))

    params = parse_args()
    tag_model = "VGG16"  # VGG16, InceptionV3, ResNet50

    # --------------------------  Reading rock image dataset  ----------------------------
    dabie = CRocks()
    dabie.IsCenterCrop = False
    dabie.y_onehot = False
    dabie.batch_size = params.batch_size
    dabie.batch_shuffle = False
    dabie.n_classes = params.image_class
    resizeFolder = params.data_folder
    taglist = ['train', 'val', 'test']
    mode_tag = params.mode_tag

    x_train_files, y_train, x_val_files, y_val, x_test_files, y_test = dabie.load_auguorresizedata(force=False,
                                                                                                   datafolder=resizeFolder,
                                                                                                   tagList=taglist,
                                                                                                   shuffle=True, x_is_images=False)
    if mode_tag == 'train':
        total_images = len(x_train_files)
    elif mode_tag == 'valid':
        total_images = len(x_val_files)
    else:
        total_images = len(x_test_files)
    n_batches = total_images // dabie.batch_size

    model_rock = RockModel_NoDense(tag_model=tag_model)

    input_shape = [None, params.image_size, params.image_size, 3]
    model_rock.build(input_shape=input_shape)
    model_rock.summary()

    weights_file = params.model_file
    if os.path.exists(weights_file):
        label_num = 24
        model_rock_2 = RockModel(label_num, params.image_size, True, tag_model=tag_model)
        input_shape = [None, params.image_size, params.image_size, 3]
        model_rock_2.build(input_shape=input_shape)

        model_rock_2.load_weights(weights_file)
        for idx, layer in enumerate(model_rock.variables):
            layer.assign(model_rock_2.variables[idx])
        print('Initializing from model_24class.h5.')

    outfile = "data/%s_%s.hdf5" % (tag_model, mode_tag)
    f = h5py.File(outfile, 'w')
    max_count = n_batches*params.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for batch in range(n_batches):
        batch_start_time = time.time()
        x_batch, y_batch = dabie.next_batch(part=mode_tag)
        images = np.array([dabie.preprocess_for_model_custom(x, params.image_size, params.image_size) for x in x_batch])

        feats = model_rock.call(images, training=False)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', (max_count, feats.shape[1]), dtype='f')
        all_feats[count:count+feats.shape[0], :] = feats.numpy()
        y_batch = y_batch.flatten()
        all_labels[count:count+feats.shape[0]] = y_batch
        count = count + feats.shape[0]

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()