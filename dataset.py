import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image, ImageOps

path = '/content/base/'

def image_preprocessing(filename, x_size, y_size):
    im = Image.open(filename)
    if filename.endswith('.png'):
        im = im.convert('RGB')

    downsampled_im = ImageOps.fit(im, (x_size, y_size), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32)/255.

    downsampled_im.close()
    im.close()

    return norm_im


if __name__ == '__main__':
    names = []

    for name in os.listdir(path):
        if name.endswith('.jpg'):
            names.append(name[:-4])

    dataset_X = np.zeros((len(names), 256, 256, 3))
    dataset_Y = np.zeros((len(names), 256, 256, 3))

    for i in range(len(names)):
        print(names[i])
        dataset_X[i] = image_preprocessing(os.path.join(path, names[i] + '.jpg'), 256, 256)
        dataset_Y[i] = image_preprocessing(os.path.join(path, names[i] + '.png'), 256, 256)

    np.save('dataset_X.npy', dataset_X)
    np.save('dataset_Y.npy', dataset_Y)
