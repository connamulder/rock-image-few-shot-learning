"""
    @Project: rock-image-transfer-learning
    @File   : 01_RockImageEpochTrain.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
"""


import os
import numpy as np
import time
import tensorflow as tf
import logging

from datasetslib.CRocks import CRocks
from rock_image_config import RockImageConfigure
from rock_cnn_model import RockModel


np.random.seed(123)
print("NumPy:{}".format(np.__version__))
print("TensorFlow:{}".format(tf.__version__))

configs = RockImageConfigure()

image_size = configs.image_size
label_num = configs.image_class

n_epochs = configs.epoch
batch_size = configs.batch_size
lr = configs.learning_rate
do_train = configs.do_train
do_Test = configs.do_test
is_scratch_train = configs.is_scratch_train

is_early_stop = configs.is_early_stop
patient = configs.patient

max_to_keep = configs.checkpoints_max_to_keep
print_per_batch = configs.print_per_batch

checkpoint_name = configs.checkpoint_name
checkpoints_dir = '%s_augu_%d' % (configs.checkpoints_dir, image_size)

input_shape_vgg = (image_size, image_size, 3)

tag_model = "VGG16"    # VGG16, InceptionV3, ResNet50
model_rock = RockModel(label_num, image_size, is_scratch_train, tag_model=tag_model)
input_shape = [None, image_size, image_size, 3]
model_rock.build(input_shape=input_shape)
model_rock.summary()


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model_R=model_rock)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

if checkpoint_manager.latest_checkpoint:
    print('Restored from {}'.format(checkpoint_manager.latest_checkpoint))
else:
    print('Initializing from scratch.')
    weights_file = os.path.join(checkpoints_dir, "model_24class.h5")
    if os.path.exists(weights_file):
        label_num = 24
        model_rock_2 = RockModel(label_num, image_size, is_scratch_train)
        input_shape = [None, image_size, image_size, 3]
        model_rock_2.build(input_shape=input_shape)

        model_rock_2.load_weights(weights_file)
        back_layer = -2
        for idx, layer in enumerate(model_rock.variables[:back_layer]):
            layer.assign(model_rock_2.variables[idx])
        label_num = configs.image_class
        print('Initializing from model_24class.h5.')


@tf.function
def train_step(x_input, y_true):
    with tf.GradientTape() as tape:
        y_pred = model_rock.call(x_input, training=True)
        train_loss = model_rock.rock_loss(y_true, y_pred)
        corrects = tf.metrics.top_k_categorical_accuracy(y_true, y_pred, 1)

        mask = tf.keras.backend.cast(
            tf.keras.backend.all(tf.keras.backend.greater(y_pred, -1e10), axis=1), tf.keras.backend.floatx())
        train_acc = tf.keras.backend.sum(corrects * mask) / tf.keras.backend.sum(mask)
    variables = model_rock.trainable_variables
    grads = tape.gradient(train_loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    return train_loss, train_acc


# --------------------------  Reading rock image dataset  ----------------------------
dabie = CRocks()
dabie.IsCenterCrop = False
dabie.y_onehot = True
dabie.batch_size = batch_size
dabie.batch_shuffle = True
dabie.n_classes = configs.image_class
resizeFolder = '%s_augu_%d' % (configs.data_folder, image_size)
taglist = ['train', 'val', 'test']

x_train_files, y_train, x_val_files, y_val, x_test_files, y_test = dabie.load_auguorresizedata(force=False,
                                                                                               datafolder=resizeFolder,
                                                                                               tagList=taglist,
                                                                                               shuffle=True, x_is_images=False)
total_images = len(x_train_files)
n_batches = total_images // dabie.batch_size
total_images_val = len(x_val_files)
n_batches_val = total_images_val // dabie.batch_size
total_images_test = len(x_test_files)
n_batches_test = total_images_test // dabie.batch_size

best_dev_loss = 10
best_dev_acc = 0
best_at_epoch = 0
unprocessed = 0
sim_weight = 0

loss_metric_mean = tf.keras.metrics.Mean()
loss_dev_metric_mean = tf.keras.metrics.Mean()
acc_metric_mean = tf.keras.metrics.Mean()
acc_dev_metric_mean = tf.keras.metrics.Mean()
acc_dev_top2_metric_mean = tf.keras.metrics.Mean()
acc_dev_top3_metric_mean = tf.keras.metrics.Mean()

summary_waite = tf.summary.create_file_writer(checkpoints_dir)
start_time = time.time()

if do_train:
    for epoch_index in range(n_epochs):
        epoch_start_time = time.time()
        print('Starting transfer learning epoch {} / {} '.format(epoch_index, n_epochs))
        dabie.reset_index()

        for batch in range(n_batches):
            batch_start_time = time.time()
            x_batch, y_batch = dabie.next_batch()
            images = np.array([dabie.preprocess_for_model_custom(x, image_size, image_size) for x in x_batch])

            loss, acc = train_step(images, y_batch)
            loss_metric_mean.update_state(float(loss))
            acc_metric_mean.update_state(float(acc))

            if batch == 0 and epoch_index == 0:
                variables = model_rock.trainable_variables
                for var in variables:
                    print(' name=%s, shape=%s' % (var.name, var.shape))
            if batch % print_per_batch == 0 and batch != 0:
                consum_time = time.time() - batch_start_time
                print('epoch %5d/%d train batch: %5d/%d, loss: %.5f, acc: %.5f, time consumption: %.3f(min)' % (
                epoch_index + 1, n_epochs, batch, n_batches, loss, acc, consum_time / 60))
        print("epoch: %d, loss: %.5f, acc: %.5f" % (epoch_index + 1, loss_metric_mean.result(), acc_metric_mean.result()))

        for batch in range(n_batches_val):
            batch_start_time = time.time()
            x_batch_val, y_batch_val = dabie.next_batch(part='valid')
            images_val = np.array([dabie.preprocess_for_model_custom(x, image_size, image_size) for x in x_batch_val])

            y_label_val = model_rock.call(images_val, training=False)
            loss_val = model_rock.rock_loss(y_batch_val, y_label_val)
            loss_dev_metric_mean.update_state(float(loss_val))

            mask = tf.keras.backend.cast(
                tf.keras.backend.all(tf.keras.backend.greater(y_label_val, -1e10), axis=1), tf.keras.backend.floatx())

            corrects = tf.metrics.top_k_categorical_accuracy(y_batch_val, y_label_val, 1)
            acc_val = tf.keras.backend.sum(corrects * mask) / tf.keras.backend.sum(mask)
            acc_dev_metric_mean.update_state((float(acc_val)))

            corrects_top2 = tf.metrics.top_k_categorical_accuracy(y_batch_val, y_label_val, 2)
            acc_val_top2 = tf.keras.backend.sum(corrects_top2 * mask) / tf.keras.backend.sum(mask)
            acc_dev_top2_metric_mean.update_state((float(acc_val_top2)))

            corrects_top3 = tf.metrics.top_k_categorical_accuracy(y_batch_val, y_label_val, 3)
            acc_val_top3 = tf.keras.backend.sum(corrects_top3 * mask) / tf.keras.backend.sum(mask)
            acc_dev_top3_metric_mean.update_state((float(acc_val_top3)))

            if batch % print_per_batch == 0 and batch != 0:
                consum_time = time.time() - batch_start_time
                print('epoch %5d/%d dev batch: %5d/%d, loss: %.5f, acc: %.5f, top2 acc: %.5f, top3 acc: %.5f, time consumption: %.3f(min)' % (
                epoch_index + 1, n_epochs, batch, n_batches_val, loss_val, acc_val, acc_val_top2, acc_val_top3, consum_time / 60))
        print("epoch: %d/%d, dev-loss: %.5f, dev-acc: %.5f, dev-top2-acc: %.5f, dev-top3-acc: %.5f" % (epoch_index + 1, n_epochs, loss_dev_metric_mean.result(),
                                                                                                       acc_dev_metric_mean.result(), acc_dev_top2_metric_mean.result(), acc_dev_top3_metric_mean.result()))
        if acc_dev_metric_mean.result() > best_dev_acc:
            unprocessed = 0
            best_dev_acc = acc_dev_metric_mean.result()
            best_at_epoch = epoch_index + 1
            checkpoint_manager.save(checkpoint_number=best_at_epoch)
            print('saved the new best model with dev-acc: %.3f' % best_dev_acc)
        else:
            unprocessed += 1

        with summary_waite.as_default():
            tf.summary.scalar('lr', lr, step=epoch_index + 1)
        with summary_waite.as_default():
            tf.summary.scalar('train-loss', loss_metric_mean.result(), step=epoch_index + 1)
        with summary_waite.as_default():
            tf.summary.scalar('dev-loss', loss_dev_metric_mean.result(), step=epoch_index + 1)
        with summary_waite.as_default():
            tf.summary.scalar('train-acc', acc_metric_mean.result(), step=epoch_index + 1)
        with summary_waite.as_default():
            tf.summary.scalar('dev-acc', acc_dev_metric_mean.result(), step=epoch_index + 1)
        with summary_waite.as_default():
            tf.summary.scalar('dev-top2-acc', acc_dev_top2_metric_mean.result(), step=epoch_index + 1)
        with summary_waite.as_default():
            tf.summary.scalar('dev-top3-acc', acc_dev_top3_metric_mean.result(), step=epoch_index + 1)
        loss_metric_mean.reset_states()
        loss_dev_metric_mean.reset_states()
        acc_metric_mean.reset_states()
        acc_dev_metric_mean.reset_states()
        acc_dev_top2_metric_mean.reset_states()
        acc_dev_top3_metric_mean.reset_states()

        if is_early_stop:
            if unprocessed > patient:
                logging.info('early stopped, no progress obtained within {} epochs'.format(patient))
                logging.info('overall best val_loss is {} at {} epoch'.format(best_dev_loss, best_at_epoch))
                logging.info('total training time consumption: %.3f(min)' % ((time.time() - start_time) / 60))
                break

if do_Test:
    loss_test_metric_mean = tf.keras.metrics.Mean()
    acc_test_metric_mean = tf.keras.metrics.Mean()
    acc_test_top2_metric_mean = tf.keras.metrics.Mean()
    acc_test_top3_metric_mean = tf.keras.metrics.Mean()

    # Saving the model parameters after training
    model_file_name = 'model_%dclass.h5' % configs.image_class
    weights_file = os.path.join(checkpoints_dir, model_file_name)
    if not os.path.exists(weights_file):
        model_rock.save_weights(weights_file)

    for batch in range(n_batches_test):
        batch_start_time = time.time()
        x_batch_test, y_batch_test = dabie.next_batch(part='test')
        images_val = np.array([dabie.preprocess_for_model_custom(x, image_size, image_size) for x in x_batch_test])

        y_label_test = model_rock.call(images_val, training=False)
        loss_test = model_rock.rock_loss(y_batch_test, y_label_test)
        loss_test_metric_mean.update_state(float(loss_test))

        mask = tf.keras.backend.cast(
            tf.keras.backend.all(tf.keras.backend.greater(y_label_test, -1e10), axis=1), tf.keras.backend.floatx())

        corrects = tf.metrics.top_k_categorical_accuracy(y_batch_test, y_label_test, 1)
        acc_test = tf.keras.backend.sum(corrects * mask) / tf.keras.backend.sum(mask)
        acc_test_metric_mean.update_state((float(acc_test)))

        corrects_top2 = tf.metrics.top_k_categorical_accuracy(y_batch_test, y_label_test, 2)
        acc_test_top2 = tf.keras.backend.sum(corrects_top2 * mask) / tf.keras.backend.sum(mask)
        acc_test_top2_metric_mean.update_state((float(acc_test_top2)))

        corrects_top3 = tf.metrics.top_k_categorical_accuracy(y_batch_test, y_label_test, 3)
        acc_test_top3 = tf.keras.backend.sum(corrects_top3 * mask) / tf.keras.backend.sum(mask)
        acc_test_top3_metric_mean.update_state((float(acc_test_top3)))
        print_per_batch = 1

        if batch % print_per_batch == 0 and batch != 0:
            consum_time = time.time() - batch_start_time
            print("test batch: %5d/%d, loss: %.5f, acc: %.5f, acc-top2: %.5f, acc-top3: %.5f, time consumption: %.3f(min)" % (batch, n_batches_val, loss_test, acc_test, acc_test_top2, acc_test_top3, consum_time / 60))
    with summary_waite.as_default():
        tf.summary.scalar('test-loss', loss_test_metric_mean.result(), step=1)
    with summary_waite.as_default():
        tf.summary.scalar('test-acc', acc_test_metric_mean.result(), step=1)
    with summary_waite.as_default():
        tf.summary.scalar('test-top2-acc', acc_test_top2_metric_mean.result(), step=1)
    with summary_waite.as_default():
        tf.summary.scalar('test-top3-acc', acc_test_top3_metric_mean.result(), step=1)
    print("test-loss: %.5f, test-acc: %.5f, test-top2-acc: %.5f, test-top3-acc: %.5f" % (loss_test_metric_mean.result(), acc_test_metric_mean.result(), acc_test_top2_metric_mean.result(), acc_test_top3_metric_mean.result()))