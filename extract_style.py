import os
import h5py

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K

import scipy.misc as img
import numpy as np
import matplotlib.pyplot as plt

# Prepare network
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

weights_path = 'vgg16_weights.h5'

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

im = img.imresize(img.imread('cat.jpg'), (224, 224)).astype(np.float32)
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

out = model.predict(im)
style_1 = out[0, :, :, :]

style_tensor = K.variable(style_1)
style_out = K.placeholder(style_1.shape)

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

p=gram_matrix(style_tensor)

fx = K.function([style_out], p)

xx = fx([style_1])

def my_gramm_matrix(x):
    features = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
    gram = features @ features.T
    return gram

img_width = 400
img_height = 400

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def my_style_loss(style, combination):
    S = my_gramm_matrix(style)
    C = my_gramm_matrix(combination)
    channels = 3
    size = img_width * img_height
    return np.sum(np.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

im1 = img.imresize(img.imread('cat.jpg'), (224, 224)).astype(np.float32)
im1 = im1.transpose((2,0,1))
im1 = np.expand_dims(im1, axis=0)

im2 = img.imresize(img.imread('cat_xxx.jpg'), (224, 224)).astype(np.float32)
im2 = im2.transpose((2,0,1))
im2 = np.expand_dims(im2, axis=0)

out = model.predict(im1)
style_im1 = out[0, :, :, :]

out = model.predict(im2)
style_im2 = out[0, :, :, :]

style_tensor_im1 = K.variable(style_im1)
style_out_im1 = K.placeholder(style_im1.shape)

style_tensor_im2 = K.variable(style_im2)
style_out_im2 = K.placeholder(style_im2.shape)

ls = style_loss(style_tensor_im1, style_tensor_im2)

fn_loss = K.function([style_out_im1, style_out_im2], ls)

xx = fn_loss([style_im1, style_im2])

yy = my_style_loss(style_im1, style_im2)
