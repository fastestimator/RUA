import os
import pdb
import random
import tempfile

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.architecture.tensorflow import UNet
from fastestimator.dataset.data import cub200
from fastestimator.op.numpyop.meta import OneOf
from fastestimator.op.numpyop.multivariate import LongestMaxSize, PadIfNeeded, ReadMat
from fastestimator.op.numpyop.numpyop import LambdaOp, NumpyOp
from fastestimator.op.numpyop.univariate import ExpandDims, ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.search import GridSearch
from fastestimator.trace.metric import Dice
from PIL import Image, ImageEnhance, ImageOps, ImageTransform


class Rescale(NumpyOp):
    def forward(self, data, state):
        return np.float32(data / 255)


class Rotate(NumpyOp):
    """ rotate between 0 to 90 degree
    """
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.degree = level * 3.0

    def forward(self, data, state):
        image, mask = data
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        degree = random.uniform(-self.degree, self.degree)
        image, mask = image.rotate(degree), mask.rotate(degree)
        image, mask = np.asarray(image), np.asarray(mask)
        return np.copy(image), np.copy(mask)


class Identity(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)


class AutoContrast(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        image = ImageOps.autocontrast(image)
        image = np.asarray(image)
        return np.copy(image), mask


class Equalize(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        image = ImageOps.equalize(image)
        image = np.asarray(image)
        return np.copy(image), mask


class Posterize(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.bit_loss_limit = level / 30 * 7

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        bits_to_keep = 8 - round(random.uniform(0, self.bit_loss_limit))
        image = ImageOps.posterize(image, bits_to_keep)
        image = np.asarray(image)
        return np.copy(image), mask


class Solarize(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_limit = level / 30 * 256

    def forward(self, data, state):
        image, mask = data
        threshold = 256 - round(random.uniform(0, self.loss_limit))
        image = np.where(image < threshold, image, 255 - image)
        return image, mask


class Sharpness(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        image = ImageEnhance.Sharpness(image).enhance(factor)
        image = np.asarray(image)
        return np.copy(image), mask


class Contrast(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        image = ImageEnhance.Contrast(image).enhance(factor)
        image = np.asarray(image)
        return np.copy(image), mask


class Color(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        image = ImageEnhance.Color(image).enhance(factor)
        image = np.asarray(image)
        return np.copy(image), mask


class Brightness(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.diff_limit = level / 30 * 0.9

    def forward(self, data, state):
        image, mask = data
        image = Image.fromarray(image)
        factor = 1.0 + random.uniform(-self.diff_limit, self.diff_limit)
        image = ImageEnhance.Brightness(image).enhance(factor)
        image = np.asarray(image)
        return np.copy(image), mask


class ShearX(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = level / 30 * 0.5

    def forward(self, data, state):
        image, mask = data
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        shear_coeff = random.uniform(-self.shear_coef, self.shear_coef)
        width, height = image.size
        xshift = round(abs(shear_coeff) * width)
        new_width = width + xshift
        image = image.transform((new_width, height),
                                ImageTransform.AffineTransform(
                                    (1.0, shear_coeff, -xshift if shear_coeff > 0 else 0.0, 0.0, 1.0, 0.0)),
                                resample=Image.BICUBIC)
        mask = mask.transform((new_width, height),
                              ImageTransform.AffineTransform(
                                  (1.0, shear_coeff, -xshift if shear_coeff > 0 else 0.0, 0.0, 1.0, 0.0)),
                              resample=Image.BICUBIC)
        image = image.resize((width, height))
        mask = mask.resize((width, height))
        image, mask = np.asarray(image), np.asarray(mask)
        return np.copy(image), np.copy(mask)


class ShearY(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = level / 30 * 0.5

    def forward(self, data, state):
        image, mask = data
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        shear_coeff = random.uniform(-self.shear_coef, self.shear_coef)
        width, height = image.size
        yshift = round(abs(shear_coeff) * height)
        newheight = height + yshift
        image = image.transform((width, newheight),
                                ImageTransform.AffineTransform(
                                    (1.0, 0.0, 0.0, shear_coeff, 1.0, -yshift if shear_coeff > 0 else 0.0)),
                                resample=Image.BICUBIC)
        mask = mask.transform((width, newheight),
                              ImageTransform.AffineTransform(
                                  (1.0, 0.0, 0.0, shear_coeff, 1.0, -yshift if shear_coeff > 0 else 0.0)),
                              resample=Image.BICUBIC)
        image = image.resize((width, height))
        mask = mask.resize((width, height))
        image, mask = np.asarray(image), np.asarray(mask)
        return np.copy(image), np.copy(mask)


class TranslateX(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level

    def forward(self, data, state):
        image, mask = data
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        width, height = image.size
        displacement = random.uniform(-self.level / 30 * height / 3, self.level / 30 * height / 3)
        image = image.transform((width, height),
                                ImageTransform.AffineTransform((1.0, 0.0, displacement, 0.0, 1.0, 0.0)),
                                resample=Image.BICUBIC)
        mask = mask.transform((width, height),
                              ImageTransform.AffineTransform((1.0, 0.0, displacement, 0.0, 1.0, 0.0)),
                              resample=Image.BICUBIC)
        image, mask = np.asarray(image), np.asarray(mask)
        return np.copy(image), np.copy(mask)


class TranslateY(NumpyOp):
    def __init__(self, level, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level

    def forward(self, data, state):
        image, mask = data
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        width, height = image.size
        displacement = random.uniform(-self.level / 30 * height / 3, self.level / 30 * height / 3)
        image = image.transform((width, height),
                                ImageTransform.AffineTransform((1.0, 0.0, 0.0, 0.0, 1.0, displacement)),
                                resample=Image.BICUBIC)
        mask = mask.transform((width, height),
                              ImageTransform.AffineTransform((1.0, 0.0, 0.0, 0.0, 1.0, displacement)),
                              resample=Image.BICUBIC)
        image, mask = np.asarray(image), np.asarray(mask)
        return np.copy(image), np.copy(mask)


def get_estimator(M, N, restore_dir, data_dir, epochs=40, batch_size=16):
    print("trying N: {}, M: {}".format(N, M))
    train_data = cub200.load_data(data_dir)
    eval_data = train_data.split(0.3, seed=42)
    aug_options = [
        Rotate(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Identity(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        AutoContrast(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Equalize(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Posterize(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Solarize(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Sharpness(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Contrast(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Color(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        Brightness(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        ShearX(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        ShearY(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        TranslateX(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train"),
        TranslateY(level=M, inputs=("image", "seg"), outputs=("image", "seg"), mode="train")
    ]
    rua_ops = [OneOf(*aug_options) for _ in range(N)]
    pipeline = fe.Pipeline(
        train_data=train_data,
        test_data=eval_data,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image", parent_path=train_data.parent_path),
            ReadMat(file='annotation', keys="seg", parent_path=train_data.parent_path),
            LongestMaxSize(max_size=256, image_in="image", mask_in="seg"),
            PadIfNeeded(min_height=256,
                        min_width=256,
                        image_in="image",
                        mask_in="seg",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0)
        ] + rua_ops + [
            Rescale(inputs="image", outputs="image"),
            ExpandDims(inputs='seg', outputs='seg'),
            LambdaOp(fn=lambda x: np.float32(x), inputs="seg", outputs="seg")
        ])
    model = fe.build(model_fn=lambda: UNet(input_size=(256, 256, 3)),
                     optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.0001))
    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs="pred"),
        CrossEntropy(inputs=("pred", "seg"), outputs="loss", form="binary"),
        UpdateOp(model=model, loss_name="loss")
    ])
    traces = [Dice(true_key="seg", pred_key="pred")]
    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=epochs, traces=traces)
    return estimator


def score_fn(search_idx, N, M, restore_dir, data_dir):
    est = get_estimator(
        N=N,
        M=M,
        restore_dir=os.path.join(restore_dir, str(search_idx)),
        data_dir=data_dir
    )
    est.fit(warmup=False)
    hist = est.test(summary="exp")
    best_acc = float(max(hist.history["test"]["Dice"].values()))
    print("Evaluated N:{} M:{}, results:{}".format(N, M, best_acc))
    return best_acc


def fastestimator_run(data_dir, restore_dir=tempfile.mkdtemp()):
    score_fn_in_use = lambda search_idx, N, M: score_fn(search_idx, N, M, restore_dir, data_dir)
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

