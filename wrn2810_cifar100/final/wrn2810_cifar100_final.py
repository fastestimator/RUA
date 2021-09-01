import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps, ImageTransform

import fastestimator as fe
from fastestimator.dataset.data import cifar100
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import OneOf, Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.search import GoldenSection
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver, RestoreWizard
from fastestimator.trace.metric import Accuracy


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


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


def get_N(level, N_max, N_min=1):
    N = level * (N_max - N_min) / 30 + N_min
    return int(N), N % 1


def get_estimator(level=22, epochs=200, batch_size=128, save_dir=tempfile.mkdtemp(), restore_dir=tempfile.mkdtemp()):
    print("trying level {}".format(level))
    # step 1: prepare dataset
    train_data, eval_data = cifar100.load_data()
    aug_options = [
        Rotate(level=level, inputs="x", outputs="x", mode="train"),
        Identity(level=level, inputs="x", outputs="x", mode="train"),
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
    N_guarantee, N_p = get_N(level, N_max=min(len(aug_options), 5))
    rua_ops = [OneOf(*aug_options) for _ in range(N_guarantee)]
    if N_p > 0:
        rua_ops.append(Sometimes(OneOf(*aug_options), prob=N_p))
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train"))
        ] + rua_ops + [
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])
    # step 2: prepare network
    model = fe.build(model_fn=lambda: WideResNet(depth=28, num_classes=100, widen_factor=10),
                     optimizer_fn=lambda x: torch.optim.SGD(x, lr=0.1, momentum=0.9, weight_decay=0.0005))

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3 prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        LRScheduler(model=model, lr_fn=lambda epoch: cosine_decay(epoch, cycle_length=epochs, init_lr=0.1)),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        RestoreWizard(directory=restore_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
