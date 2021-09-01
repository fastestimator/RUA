import os
import random
import tempfile

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageTransform
from tensorflow.keras import Model, layers

import fastestimator as fe
from fastestimator.dataset.data.svhn_cropped import load_data
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import OneOf
from fastestimator.op.numpyop.univariate import CoarseDropout
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.search import GridSearch
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import RestoreWizard
from fastestimator.trace.metric import Accuracy


def WideResidualNetwork(input_shape, depth=28, width=8, dropout_rate=0.0, classes=10, activation='softmax'):

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)' 'should be divisible by 6.')

    img_input = layers.Input(shape=input_shape)

    x = __create_wide_residual_network(classes, img_input, True, depth, width, dropout_rate, activation)
    inputs = img_input
    # Create model.
    model = Model(inputs, x)
    return model


def __conv1_block(inputs):
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def __conv2_block(inputs, k=1, dropout=0.0):
    init = inputs

    # Check if input number of filters is same as 16 * k, else create
    # convolution2d for this input
    if init.shape[-1] != 16 * k:
        init = layers.Conv2D(16 * k, (1, 1), activation='linear', padding='same')(init)

    x = layers.Conv2D(16 * k, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(16 * k, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    m = init + x
    return m


def __conv3_block(inputs, k=1, dropout=0.0):
    init = inputs
    # Check if input number of filters is same as 32 * k, else
    # create convolution2d for this input
    if init.shape[-1] != 32 * k:
        init = layers.Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)
    x = layers.Conv2D(32 * k, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(32 * k, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    m = init + x
    return m


def ___conv4_block(inputs, k=1, dropout=0.0):
    init = inputs
    if init.shape[-1] != 64 * k:
        init = layers.Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)
    x = layers.Conv2D(64 * k, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(64 * k, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    m = init + x
    return m


def __create_wide_residual_network(nb_classes,
                                   img_input,
                                   include_top,
                                   depth=28,
                                   width=8,
                                   dropout=0.0,
                                   activation='softmax'):
    ''' Creates a Wide Residual Network with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Compute N = (n - 4) / 6.
               For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
               For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
               For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        width: Width of the network.
        dropout: Adds dropout if value is greater than 0.0
    Returns:a Keras Model
    '''
    N = (depth - 4) // 6
    x = __conv1_block(img_input)
    nb_conv = 4
    for i in range(N):
        x = __conv2_block(x, width, dropout)
        nb_conv += 2

    x = layers.MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = __conv3_block(x, width, dropout)
        nb_conv += 2

    x = layers.MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = ___conv4_block(x, width, dropout)
        nb_conv += 2

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(nb_classes, activation=activation)(x)
    return x


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


class Identity(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)


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


class Scale(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        data = data / 255
        return data


def get_estimator(N, M, epochs=200, batch_size=128, restore_dir=tempfile.mkdtemp()):
    print("trying N:{}, M: {}".format(N, M))
    # step 1: prepare dataset
    train_ds, test_ds = load_data()
    aug_options = [
        Rotate(level=M, inputs="x", outputs="x", mode="train"),
        Identity(level=M, inputs="x", outputs="x", mode="train"),
        AutoContrast(level=M, inputs="x", outputs="x", mode="train"),
        Equalize(level=M, inputs="x", outputs="x", mode="train"),
        Posterize(level=M, inputs="x", outputs="x", mode="train"),
        Solarize(level=M, inputs="x", outputs="x", mode="train"),
        Sharpness(level=M, inputs="x", outputs="x", mode="train"),
        Contrast(level=M, inputs="x", outputs="x", mode="train"),
        Color(level=M, inputs="x", outputs="x", mode="train"),
        Brightness(level=M, inputs="x", outputs="x", mode="train"),
        ShearX(level=M, inputs="x", outputs="x", mode="train"),
        ShearY(level=M, inputs="x", outputs="x", mode="train"),
        TranslateX(level=M, inputs="x", outputs="x", mode="train"),
        TranslateY(level=M, inputs="x", outputs="x", mode="train")
    ]
    rua_ops = [OneOf(*aug_options) for _ in range(N)]
    pipeline = fe.Pipeline(
        train_data=train_ds,
        test_data=test_ds,
        batch_size=batch_size,
        ops=rua_ops +
        [Scale(inputs="x", outputs="x"), CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1)])
    # step 2: prepare network
    model = fe.build(model_fn=lambda: WideResidualNetwork(input_shape=(32, 32, 3), depth=28, width=2),
                     optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3 prepare estimator
    traces = [
        LRScheduler(model=model, lr_fn=lambda epoch: cosine_decay(epoch, cycle_length=epochs, init_lr=1e-3)),
        Accuracy(true_key="y", pred_key="y_pred"),
        RestoreWizard(directory=restore_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


def score_fn(search_idx, N, M, restore_dir):
    est = get_estimator(N=N, M=M, restore_dir=os.path.join(restore_dir, str(search_idx)))
    est.fit(warmup=False)
    hist = est.test(summary="exp")
    best_acc = float(max(hist.history["test"]["accuracy"].values()))
    print("Evaluated N:{} M:{}, results:{}".format(N, M, best_acc))
    return best_acc


def fastestimator_run(restore_dir=tempfile.mkdtemp()):
    restore_dir = os.path.join(restore_dir, "svhn")
    score_fn_in_use = lambda search_idx, N, M: score_fn(search_idx, N, M, restore_dir=restore_dir)
    gss = GridSearch(score_fn=score_fn_in_use,
                     params={
                         "N": [x + 1 for x in range(10)], "M": [3 * (x + 1) for x in range(10)]
                     })
    gss.fit(save_dir=restore_dir)
    print("search history:")
    print(gss.get_search_results())
    print("=======================")
    print("best result:")
    print(gss.get_best_results())
