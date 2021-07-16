import math
import os
import random
import tempfile

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.dataset import LabeledDirDataset
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import CenterCrop, HorizontalFlip, RandomResizedCrop, SmallestMaxSize
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.schedule import EpochScheduler
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util import get_num_devices
from PIL import Image, ImageEnhance, ImageOps, ImageTransform
from tensorflow.keras import layers


class Rotate(NumpyOp):
    """ rotate between 0 to 90 degree
    """
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.degree = level * 3.0

    def forward(self, data, state):
        im = Image.fromarray(data)
        degree = random.uniform(-self.degree, self.degree)
        im = im.rotate(degree)
        return np.copy(np.asarray(im))


class AutoContrast(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        im = Image.fromarray(data)
        im = ImageOps.autocontrast(im)
        return np.copy(np.asarray(im))


class Equalize(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        im = Image.fromarray(data)
        im = ImageOps.equalize(im)
        return np.copy(np.asarray(im))


class Posterize(NumpyOp):
    # resuce the number of bits for each channel, this may be inconsistent with original implementation
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.bit_loss_limit = level / 30 * 7

    def forward(self, data, state):
        im = Image.fromarray(data)
        bits_to_keep = 8 - round(random.uniform(0, self.bit_loss_limit))
        im = ImageOps.posterize(im, bits_to_keep)
        return np.copy(np.asarray(im))


class Solarize(NumpyOp):
    # this may be inconsistent with original implementation
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_limit = level / 30 * 256

    def forward(self, data, state):
        threshold = 256 - round(random.uniform(0, self.loss_limit))
        data = np.where(data < threshold, data, 255 - data)
        return data


class Sharpness(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        im = ImageEnhance.Sharpness(im).enhance(factor)
        return np.copy(np.asarray(im))


class Contrast(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        im = ImageEnhance.Contrast(im).enhance(factor)
        return np.copy(np.asarray(im))


class Color(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        im = ImageEnhance.Color(im).enhance(factor)
        return np.copy(np.asarray(im))


class Brightness(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        im = Image.fromarray(data)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        im = ImageEnhance.Brightness(im).enhance(factor)
        return np.copy(np.asarray(im))


class ShearX(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = level / 30 * 0.5

    def forward(self, data, state):
        im = Image.fromarray(data)
        shear_coeff = random.uniform(-self.shear_coef, self.shear_coef)
        width, height = im.size
        xshift = round(abs(shear_coeff) * width)
        new_width = width + xshift
        im = im.transform((new_width, height),
                          ImageTransform.AffineTransform(
                              (1.0, shear_coeff, -xshift if shear_coeff > 0 else 0.0, 0.0, 1.0, 0.0)),
                          resample=Image.BICUBIC)
        im = im.resize((width, height))
        return np.copy(np.asarray(im))


class ShearY(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = level / 30 * 0.5

    def forward(self, data, state):
        im = Image.fromarray(data)
        shear_coeff = random.uniform(-self.shear_coef, self.shear_coef)
        width, height = im.size
        yshift = round(abs(shear_coeff) * height)
        newheight = height + yshift
        im = im.transform((width, newheight),
                          ImageTransform.AffineTransform(
                              (1.0, 0.0, 0.0, shear_coeff, 1.0, -yshift if shear_coeff > 0 else 0.0)),
                          resample=Image.BICUBIC)
        im = im.resize((width, height))
        return np.copy(np.asarray(im))


class TranslateX(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level

    def forward(self, data, state):
        im = Image.fromarray(data)
        width, height = im.size
        displacement = random.uniform(-self.level / 30 * height / 3, self.level / 30 * height / 3)
        im = im.transform((width, height),
                          ImageTransform.AffineTransform((1.0, 0.0, displacement, 0.0, 1.0, 0.0)),
                          resample=Image.BICUBIC)
        return np.copy(np.asarray(im))


class TranslateY(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level

    def forward(self, data, state):
        im = Image.fromarray(data)
        width, height = im.size
        displacement = random.uniform(-self.level / 30 * height / 3, self.level / 30 * height / 3)
        im = im.transform((width, height),
                          ImageTransform.AffineTransform((1.0, 0.0, 0.0, 0.0, 1.0, displacement)),
                          resample=Image.BICUBIC)
        return np.copy(np.asarray(im))


def center_crop(data):
    image_height, image_width, _ = data.shape
    padded_center_crop_size = int(224 / (224 + 32) * min(image_height, image_height))
    y1 = ((image_height - padded_center_crop_size) + 1) // 2
    x1 = ((image_width - padded_center_crop_size) + 1) // 2
    return data[y1:y1 + padded_center_crop_size, x1:x1 + padded_center_crop_size, :]


class RGBScale(fe.op.tensorop.TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.mean_rgb = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], shape=[1, 1, 3], dtype=tf.float32)

    def forward(self, data, state):
        image = tf.cast(data, tf.float32)
        return image - self.mean_rgb


def lr_warmup_fn(step, init_lr, steps_per_epoch, warmup_epochs=10):
    warmup_steps = steps_per_epoch * warmup_epochs
    return init_lr * step / warmup_steps


def lr_decay_fn(epoch, init_lr):
    if epoch <= 60:
        lr = init_lr
    elif epoch <= 120:
        lr = init_lr * 0.1
    elif epoch <= 160:
        lr = init_lr * 0.01
    else:
        lr = init_lr * 0.001
    return lr


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.L2(l2_weight_decay) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = layers.Conv2D(filters1, (1, 1),
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,
                      kernel_size,
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon,
                                  name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,
                      kernel_size,
                      strides=strides,
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon,
                                  name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1),
                             strides=strides,
                             use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=batch_norm_decay,
                                         epsilon=batch_norm_epsilon,
                                         name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(use_l2_regularizer=True, batch_norm_decay=0.9, batch_norm_epsilon=1e-5):
    block_config = dict(use_l2_regularizer=use_l2_regularizer,
                        batch_norm_decay=batch_norm_decay,
                        batch_norm_epsilon=batch_norm_epsilon)
    img_input = layers.Input(shape=(224, 224, 3))
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name='conv1')(x)
    x = layers.BatchNormalization(momentum=batch_norm_decay, epsilon=batch_norm_epsilon, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', **block_config)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', **block_config)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', **block_config)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000,
                     kernel_initializer=tf.compat.v1.keras.initializers.random_normal(stddev=0.01),
                     kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     name='fc1000')(x)
    # A softmax that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)
    return tf.keras.Model(img_input, x, name='resnet50')


def get_probability(N, P):
    """
    N: number of augmentation
    P: the probability of having at least 1 augmentation

    return: the N needed
    """
    return 1 - math.exp(math.log(1 - P) / N)


def get_estimator(level, data_dir, batch_per_gpu=256, epochs=180, save_dir=tempfile.mkdtemp()):
    print("trying level {}".format(level))
    train_ds = LabeledDirDataset(os.path.join(data_dir, "train"))
    eval_ds = LabeledDirDataset(os.path.join(data_dir, "val"))
    batch_size = batch_per_gpu * get_num_devices()
    aug_options = [
        Rotate(level=level, inputs="x", outputs="x", mode="train"),
        AutoContrast(level=level, inputs="x", outputs="x", mode="train"),
        Equalize(level=level, inputs="x", outputs="x", mode="train"),
        Posterize(level=level, inputs="x", outputs="x", mode="train"),
        Solarize(level=level, inputs="x", outputs="x", mode="train"),
        Sharpness(level=level, inputs="x", outputs="x", mode="train"),
        Contrast(level=level, inputs="x", outputs="x", mode="train"),
        Color(level=level, inputs="x", outputs="x", mode="train"),
        Brightness(level=level, inputs="x", outputs="x", mode="train"),
        ShearX(level=level, inputs="x", outputs="x", mode="train"),
        ShearY(level=level, inputs="x", outputs="x", mode="train"),
        TranslateX(level=level, inputs="x", outputs="x", mode="train"),
        TranslateY(level=level, inputs="x", outputs="x", mode="train")
    ]
    aug_ops = [Sometimes(op, prob=get_probability(len(aug_options), 0.99)) for op in aug_options]
    pipeline = Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            RandomResizedCrop(height=224,
                              width=224,
                              image_in="x",
                              image_out="x",
                              interpolation=cv2.INTER_CUBIC,
                              scale=(0.05, 1.0),
                              mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train"))
        ] + aug_ops + [
            SmallestMaxSize(max_size=256, image_in="x", image_out="x", interpolation=cv2.INTER_CUBIC, mode="eval"),
            CenterCrop(height=224, width=224, image_in="x", image_out="x", mode="eval")
        ])
    # step 2
    init_lr = 0.1 * batch_size / 256
    model = fe.build(model_fn=resnet50,
                     optimizer_fn=lambda: tf.optimizers.SGD(learning_rate=init_lr, momentum=0.9),
                     mixed_precision=True)
    network = fe.Network(ops=[
        RGBScale(inputs="x", outputs="x"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    lr_schedule = {
        1:
        LRScheduler(
            model=model,
            lr_fn=lambda step: lr_warmup_fn(step, init_lr=init_lr, steps_per_epoch=np.ceil(len(train_ds) / batch_size))
        ),
        11:
        LRScheduler(model=model, lr_fn=lambda epoch: lr_decay_fn(epoch, init_lr=init_lr))
    }
    traces.append(EpochScheduler(lr_schedule))
    # step 3
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


def evaluate_result(data_dir, level):
    est = get_estimator(data_dir=data_dir, level=level)
    hist = est.fit(summary="exp", warmup=False)
    best_acc = float(max(hist.history["eval"]["max_accuracy"].values()))
    print("Evaluated level {}, results:{}".format(level, best_acc))
    return best_acc


def gss(data_dir, a, b, total_trial=10):
    results = {}
    h = b - a
    invphi = (math.sqrt(5) - 1) / 2
    invphi2 = (3 - math.sqrt(5)) / 2
    c = int(a + invphi2 * h)
    d = int(a + invphi * h)
    yc = evaluate_result(data_dir, level=c)
    results[c] = yc
    yd = evaluate_result(data_dir, level=d)
    results[d] = yd
    for i in range(total_trial - 2):
        if yc > yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = int(a + invphi2 * h)
            if c in results:
                yc = results[c]
            else:
                yc = evaluate_result(data_dir, level=c)
                results[c] = yc
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = int(a + invphi * h)
            if d in results:
                yd = results[d]
            else:
                yd = evaluate_result(data_dir, level=d)
                results[d] = yd
    max_value_keys = [key for key in results.keys() if results[key] == max(results.values())]
    return max_value_keys[0], results[max_value_keys[0]]


def fastestimator_run(data_dir):
    best_level, best_acc = gss(data_dir, a=1, b=30, total_trial=7)
    print("best level is {}, best accuracy is {}".format(best_level, best_acc))
