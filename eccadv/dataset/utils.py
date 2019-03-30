from PIL import Image
import numpy as np


def dump_image(fpath, arr: np.array):
    w, h, c = arr.shape
    if c == 1:
        mode = "I"
        arr = arr.squeeze()
    else:
        mode = "RGB"
    Image.fromarray(arr, mode).save(fpath)
