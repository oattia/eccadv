import itertools

import numpy as np


def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum([int(s1[i] != s2[i]) for i in range(len(s1))])


def bit_list_to_str(bl):
    return "".join([str(int(bit)) for bit in bl])


def int_to_bit_str(i, length):
    return "{num:0{len}b}".format(**{"num": int(i), "len": int(length)})


def all_bit_strings(length):
    for seq in itertools.product("01", repeat=length):
        yield "".join(seq)


def pop_count(s):
    return sum([1 if bit == "1" else 0 for bit in s])


def gray_code(i):
    return i ^ (i >> 1)


def n_bits_with_k_set(n, k):
    for bits in itertools.combinations(range(n), k):
        s = ["0"] * n
        for bit in bits:
            s[bit] = "1"
        yield "".join(s)


def flip_random(s, k):
    idxs = np.random.choice(len(s), k, replace=False)
    return bit_list_to_str([int(bit) ^ int(i in idxs) for i, bit in enumerate(s)])


def encode_probs(s, e, factor):
    ps = np.exp(-1 * factor * np.array(list(range(e+1))))
    ps = ps / np.sum(ps)
    k = np.random.choice(e+1, 1, replace=False, p=ps)
    return flip_random(s, k)
