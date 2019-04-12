from PIL import Image


def dump_image(arr, fpath):
    w, h, c = arr.shape
    if c == 1:
        mode = "I"
        arr = arr.squeeze()
    else:
        mode = "RGB"
    Image.fromarray(arr, mode).save(fpath)
