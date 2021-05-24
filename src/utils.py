import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from model import Vgg16

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = tf.constant(img, dtype = tf.float64)
    return img


def tensor_save_rgbimage(tensor, filename):
    img = tf.constant(tensor, dtype = tf.unit8)
    img = tf.clip_by_value(img, clip_value_min = 0, clip_value_max = 255)
    img = tf.transpose(img, perm = [1, 2, 0])
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename):
    # Conver BGR to RGB
    (b, g, r) = tf.split(tensor, num_or_size_splits = 3, axis = 3)
    tensor = tf.concat((r, g, b), axis = 3)
    tensor_save_rgbimage(tensor, filename)


# def gram_matrix(y):
#     (b, ch, h, w) = y.get_shape()        # b - Batch_size, ch- Channels, h - Height, w - Width
#     features = tf.reshape(y,[b, ch, w * h])
#     features_t = tf.transpose(feature, perm = [1, 2])
#     gram = tf.matmul(features, features_t) / (ch * h * w)
#     return gram

def gram_matrix(x):                                
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def subtract_imagenet_mean_batch(X):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    b, ch, w, c = X.shape
    X[:, 0, :, :] -= 103.939
    X[:, 1, :, :] -= 116.779
    X[:, 2, :, :] -= 123.680
    return X


def add_imagenet_mean_batch(X):
    """Add ImageNet mean pixel-wise from a BGR image."""
    b, ch, w, c = X.shape
    X[:, 0, :, :] += 103.939
    X[:, 1, :, :] += 116.779
    X[:, 2, :, :] += 123.680
    return X

def imagenet_clamp_batch(batch, low, high):
    batch[:,0,:,:] = tf.clip_by_value(batch[:,0,:,:], clip_value_min = low-103.939, clip_value_max = high-103.939)
    batch[:,1,:,:] = tf.clip_by_value(batch[:,1,:,:], clip_value_min = low-103.939, clip_value_max = high-103.939)
    batch[:,2,:,:] = tf.clip_by_value(batch[:,2,:,:], clip_value_min = low-103.939, clip_value_max = high-103.939)

def preprocess_batch(batch):
    batch = tf.transpose(img, perm = [0, 1])
    (r, g, b) = tf.split(batch, num_or_size_splits = 3, axis = 3)
    batch = tf.concat((b, g, r), axis = 3)
    batch = tf.transpose(img, perm = [0, 1])
    return batch


# def init_vgg16(model_folder):
#     """load the vgg16 model feature"""
#     if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
#         if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
#             os.system(
#                 'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
#         vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
#         vgg = Vgg16()
#         for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
#             dst.data[:] = src
#         torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))


class StyleLoader():
    def __init__(self, style_folder, style_size, cuda=True):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
    
    def get(self, i):
        idx = i%len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.style_size)    
        style = style.unsqueeze(0)
        style = preprocess_batch(style)
        style_v = tf.Variable(style)
        return style_v

    def size(self):
        return len(self.files)