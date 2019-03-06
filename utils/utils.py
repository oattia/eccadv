import itertools


def bit_list_to_str(bl):
    return "".join([str(int(bit)) for bit in bl])


def int_to_bit_str(i, length):
    return "{num:0{len}b}".format(**{"num": int(i), "len": int(length)})


def all_bit_strings(length):
    return ["".join(seq) for seq in itertools.product("01", repeat=length)]


def gray_code(i):
    return i ^ (i >> 1)
