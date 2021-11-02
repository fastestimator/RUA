import os
import pdb
import random
import tempfile

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.dataset import LabeledDirDataset
from fastestimator.op.numpyop.meta import OneOf
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, cosine_decay
from fastestimator.search import GridSearch
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.metric import Accuracy
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


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output


def point_wise_feed_forward_network(em_dim, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(em_dim)  # (batch_size, seq_len, em_dim)
    ])


class MultiHeadAttention(layers.Layer):
    def __init__(self, em_dim, num_heads):
        super().__init__()
        assert em_dim % num_heads == 0, "model dimension must be multiple of number of heads"
        self.num_heads = num_heads
        self.em_dim = em_dim
        self.depth = em_dim // self.num_heads
        self.wq = layers.Dense(em_dim)
        self.wk = layers.Dense(em_dim)
        self.wv = layers.Dense(em_dim)
        self.dense = layers.Dense(em_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # B, num_heads, seq_len, depth

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # B, seq_len, em_dim
        k = self.wk(k)  # B, seq_len, em_dim
        v = self.wv(v)  # B, seq_len, em_dim
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  #B, seq_len, num_heads, depth
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.em_dim))  # B, seq_len, em_dim
        output = self.dense(concat_attention)
        return output


class EncoderLayer(layers.Layer):
    def __init__(self, em_dim, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(em_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(em_dim, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(layers.Layer):
    def __init__(self, num_layers, em_dim, num_heads, dff, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(em_dim, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        return x


class PositionEmbedding(layers.Layer):
    def __init__(self, image_size, patch_size, em_dim):
        super().__init__()
        h, w, _ = image_size
        assert h % patch_size == 0 and w % patch_size == 0, "image size must be an integer multiple of patch size"
        self.position_embedding = tf.Variable(tf.zeros(shape=(1, h * w // patch_size**2 + 1, em_dim)),
                                              trainable=True,
                                              name="position_embedding")

    def call(self, x):
        return x + self.position_embedding


class ClsToken(layers.Layer):
    def __init__(self, em_dim):
        super().__init__()
        self.cls_token = tf.Variable(tf.zeros(shape=(1, 1, em_dim)), trainable=True, name="cls_token")
        self.em_dim = em_dim

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.concat([tf.broadcast_to(self.cls_token, (batch_size, 1, self.em_dim)), x], axis=1)


def transformer_encoder(image_size, patch_size=16, num_layers=12, em_dim=768, num_heads=12, dff=3072, rate=0.1):
    inputs = layers.Input(shape=image_size)
    # Patch Embedding
    x = layers.Conv2D(em_dim, kernel_size=patch_size, strides=patch_size, use_bias=False)(inputs)  #[B, H, W, em_dim]
    x = layers.Reshape((-1, em_dim))(x)  # [B, num_patches, em_dim]
    x = ClsToken(em_dim)(x)  # [B, num_patches + 1, em_dim]
    x = PositionEmbedding(image_size, patch_size, em_dim)(x)
    x = Encoder(num_layers=num_layers, em_dim=em_dim, num_heads=num_heads, dff=dff, rate=rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])  # only need the embedding w.r.t [cls] token
    return tf.keras.Model(inputs=inputs, outputs=x)


def vision_transformer(num_class,
                       image_size,
                       patch_size=16,
                       num_layers=12,
                       em_dim=768,
                       num_heads=12,
                       dff=3072,
                       rate=0.1):
    inputs = layers.Input(shape=image_size)
    backbone = transformer_encoder(image_size, patch_size, num_layers, em_dim, num_heads, dff, rate)
    x = backbone(inputs)
    x = layers.Dense(num_class)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


class Rescale(NumpyOp):
    def forward(self, data, state):
        return np.float32(data / 255)


def lr_schedule_warmup(step, train_steps_epoch, init_lr):
    warmup_steps = train_steps_epoch * 5
    if step < warmup_steps:
        lr = init_lr / warmup_steps * step
    else:
        lr = init_lr
    return lr


def get_estimator(N, M, init_lr=0.1, batch_size=512, epochs=100, data_dir="/data/shared_data/tiny-imagenet-200"):
    print("trying N: {}, M: {}".format(N, M))
    train_data = LabeledDirDataset(os.path.join(data_dir, "train"))
    test_data = LabeledDirDataset(os.path.join(data_dir, "val"))
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
    pipeline = fe.Pipeline(train_data=train_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ReadImage(inputs="x", outputs="x")] + rua_ops + [Rescale(inputs="x", outputs="x")])
    model = fe.build(
        model_fn=lambda: vision_transformer(
            num_class=200, image_size=(64, 64, 3), patch_size=4, num_layers=6, em_dim=256, num_heads=8, dff=512),
        optimizer_fn=lambda: tf.optimizers.SGD(init_lr, momentum=0.9))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    lr_schedule = {
        1:
        LRScheduler(
            model=model,
            lr_fn=lambda step: lr_schedule_warmup(
                step, train_steps_epoch=np.ceil(len(train_data) / batch_size), init_lr=init_lr)),
        6:
        LRScheduler(
            model=model,
            lr_fn=lambda epoch: cosine_decay(
                epoch, cycle_length=epochs - 5, init_lr=init_lr, min_lr=init_lr / 100, start=6))
    }
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=[Accuracy(true_key="y", pred_key="y_pred"), EpochScheduler(lr_schedule)])
    return estimator


def score_fn(search_idx, N, M):
    est = get_estimator(N=N, M=M)
    est.fit(warmup=False)
    hist = est.test(summary="exp")
    best_acc = float(max(hist.history["test"]["accuracy"].values()))
    print("Evaluated N:{} M:{}, results:{}".format(N, M, best_acc))
    return best_acc


def fastestimator_run(restore_dir=tempfile.mkdtemp()):
    restore_dir = os.path.join(restore_dir, "svhn")
    score_fn_in_use = lambda search_idx, N, M: score_fn(search_idx, N, M)
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
