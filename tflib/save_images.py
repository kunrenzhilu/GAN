"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path, iteration=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples/rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh+nh, w*nw+nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh+nh, w*nw+nw))

    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw
        if iteration is not None and iteration % 19999 == 0:
            save_images.counter += 1
            imsave('output/' + '{}.png'.format(save_images.counter), x)
        img[j*h+j:j*h+h+j, i*w+i:i*w+w+i] = x


    imsave(save_path, img)
save_images.counter = 0