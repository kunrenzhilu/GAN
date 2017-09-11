from scipy import stats as sp
import numpy as np
import os
from PIL import Image


def get_frequency_map(path, smoothing=False):
    file_names = os.listdir(path)
    map_shape = np.array(Image.open(path + file_names[0])).shape[:2]
    if smoothing:
        map = np.ones(map_shape)    # ones for smoothing
    else:
        map = np.zeros(map_shape)
    for fn in file_names:
        im = np.array(Image.open(path + fn))
        (r, c) = map_shape
        for i in range(r):
            for j in range(c):
                if im[i][j][0] >= 200 and im[i][j][1] <=50 and im[i][j][2] <= 50:
                    map[i][j] += 1
    return map
    # print(map[29:35, 29:35])

input_map = get_frequency_map("images/", smoothing=False).flatten()
output_map = get_frequency_map("output/", smoothing=True).flatten()
KL = sp.entropy(input_map, output_map, 2)
print(KL)

output_map = np.ones(64 * 64)
KL = sp.entropy(input_map, output_map, 2)
print(KL)


