from scipy import misc


def dump_image(arr, fpath):
    w, h, c = arr.shape
    if c == 1:
        arr = arr.squeeze(axis=2)
    misc.imsave(fpath, arr)
