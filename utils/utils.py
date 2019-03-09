import itertools

import numpy as np


def bit_list_to_str(bl):
    return "".join([str(int(bit)) for bit in bl])


def int_to_bit_str(i, length):
    return "{num:0{len}b}".format(**{"num": int(i), "len": int(length)})


def all_bit_strings(length):
    return ["".join(seq) for seq in itertools.product("01", repeat=length)]


def gray_code(i):
    return i ^ (i >> 1)


def flip_random(s, k):
    idxs = np.random.choice(len(s), k, replace=False)
    return bit_list_to_str([int(bit) ^ int(i in idxs) for i, bit in enumerate(s)])


def encode_probs(s, e, factor=1):
    ps = np.exp(-1 * factor * np.array(list(range(e+1))))
    ps = ps / np.sum(ps)
    k = np.random.choice(e+1, 1, replace=False, p=ps)
    return flip_random(s, k)
