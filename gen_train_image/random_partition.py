from shutil import move
import os
from math import ceil
from random import shuffle

file_names = os.listdir("images/")
shuffle(file_names)
train_size = ceil(0.8 * len(file_names))
valid_size = len(file_names) - train_size
counter = 0
for fn in file_names:
    if counter < train_size :
        move("images/" + fn, "train_64x64/" + fn)
    else:
        move("images/" + fn, "valid_64x64/" + fn)
    counter += 1