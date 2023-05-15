"""
    @Project: rock-image-transfer-learning
    @File   : rock_cnn_model.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
"""


import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


# -----------------------  Define the model of rock image transfer learning ----------------------------
class RockModel(tf.keras.Model):
    def __init__(self, num_classes, image_size, is_scratch_train, tag_model="VGG16", *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.input_shape_vgg = (image_size, image_size, 3)
        if tag_model == "VGG16":
            self.base_model = VGG16(input_shape=self.input_shape_vgg, weights='imagenet', include_top=False)
        elif tag_model == "InceptionV3":
            self.base_model = InceptionV3(input_shape=self.input_shape_vgg, weights='imagenet', include_top=False)
        elif tag_model == "ResNet50":
            self.base_model = ResNet50(input_shape=self.input_shape_vgg, weights='imagenet', include_top=False)
        for layer in self.base_model.layers:
            layer.trainable = is_scratch_train

        self.pool = GlobalAveragePooling2D()
        self.dense = Dense(num_classes, activation='softmax', name="rock_dense")

    @tf.function
    def call(self, inputs, training=None):
        output_layer = self.base_model(inputs)
        output_layer = self.pool(output_layer)
        output_logits = self.dense(output_layer)

        return output_logits

    @tf.function
    def rock_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        loss = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_mean(loss)

        return loss


# -----------------------  Define the rock image model without fully connection layer ----------------------------
class RockModel_NoDense(tf.keras.Model):
    def __init__(self, image_size=224, is_scratch_train=False, tag_model="VGG16", *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.input_shape_vgg = (image_size, image_size, 3)
        if tag_model == "VGG16":
            self.base_model = VGG16(input_shape=self.input_shape_vgg, weights='imagenet', include_top=False)
        elif tag_model == "InceptionV3":
            self.base_model = InceptionV3(input_shape=self.input_shape_vgg, weights='imagenet', include_top=False)
        elif tag_model == "ResNet50":
            self.base_model = ResNet50(input_shape=self.input_shape_vgg, weights='imagenet', include_top=False)
        for layer in self.base_model.layers:
            layer.trainable = is_scratch_train

        self.pool = GlobalAveragePooling2D()

    # @tf.function
    def call(self, inputs, training=None):
        output_layer = self.base_model(inputs)
        output_layer = self.pool(output_layer)

        return output_layer


if __name__ == '__main__':
    from rock_image_config import RockImageConfigure

    configs = RockImageConfigure()
    num_label = configs.image_class
    image_size = configs.image_size
    is_train_scratch = configs.is_scratch_train
    tag_model = "VGG16"    # VGG16, InceptionV3, ResNet50

    model_rock = RockModel(num_label, image_size, is_train_scratch, tag_model=tag_model)
    input_shape = [None, image_size, image_size, 3]
    model_rock.build(input_shape=input_shape)
    model_rock.summary()

    print("\nOutput of Rock_CNN_Model variables: ")
    variables = model_rock.trainable_variables
    for var in variables:
        print(' name=%s, shape=%s' % (var.name, var.shape))

    model_rock_Nodense = RockModel_NoDense(image_size, is_train_scratch, tag_model=tag_model)
    input_shape = [None, image_size, image_size, 3]
    model_rock_Nodense.build(input_shape=input_shape)
    model_rock_Nodense.summary()