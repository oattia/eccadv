from coders.channel.channel_coder import ChannelCoder
from coders.utils import encode_probs, int_to_bit_str

import unireedsolomon as urs


class ReedSolomonCoder(ChannelCoder):
    """
    Encodes binary strings by using Reed-Solomon algorithm.
    Encodes every single bit in the source coder output to a whole byte.
    Can correct (n - k) / 2 errors, where k is the input size, and n is output size, in bytes.
    """
    def __init__(self, name, prob, factor, n):
        super(ReedSolomonCoder, self).__init__(name, prob, factor)
        self.n = n  # output_size, in bytes
        self.k = None
        self.max_correct = None
        self.rs = None

    def set_source_coder(self, source_coder):
        super(ReedSolomonCoder, self).set_source_coder(source_coder)
        self.k = self.source_coder.output_size
        self.max_correct = (self.n - self.k) // 2
        assert self.max_correct >= 1
        self.rs = urs.RSCoder(self.n, self.k)

    def encode(self, label):
        assert self.is_set()
        bs = self.source_coder.encode(label)
        encoded_bytes = [int(x) for x in self.rs.encode([int(s) for s in bs], return_string=False)]
        encoding = "".join([int_to_bit_str(encoded_byte, 8) for encoded_byte in encoded_bytes])
        if self.prob:
            encoding = encode_probs(encoding, self.max_correct, self.factor)
        return encoding

    def decode(self, bs):
        try:
            assert self.is_set()
            bs = "".join([str(x) for x in bs])
            encoded_bytes = [int(bs[i:i + 8], base=2) for i in range(0, len(bs), 8)]
            decoded_bytes, _ = self.rs.decode(encoded_bytes, return_string=False, nostrip=True)
            decoding = "".join([str(bit) for bit in decoded_bytes])
            return self.source_coder.decode(decoding)
        except Exception as e:
            return "{} because {}".format(ChannelCoder.CANT_DECODE, str(e))

    def output_size(self):
        return self.n * 8


if __name__ == "__main__":
    n = 200
    k = 15
    d = (n - k + 1)
    q = 1 << 8

    # rs = urs.RSCoder(n, k)
    # import scipy.special as ss
    #
    # print(ss.comb(n, d) * (q - 1))
    # print(rs.encode([11173], return_string=False))
    #
    # # for g in sorted(rs.g.items(), key=lambda c: c[0]):
    # #     print(g)
    # b = [0] * (k-1) + [1] + [173] * (d-1)
    # print(b)
    # print(len(rs.g[k].coefficients))
    # print(rs.check(b))
    # print(rs.decode(b, return_string=False)[0])
    # import sys; sys.exit(0)
    # alpha = list(range(1 << k))
    # from coders.source.source_coder import DummySourceCoder
    # r = ReedSolomonCoder("rsc", prob=False, n=n, factor=0)
    # from coders.utils import all_bit_strings, pop_count, n_bits_with_k_set
    # from collections import defaultdict, OrderedDict
    # count = 0
    # for loop, bs in enumerate(n_bits_with_k_set(n * 8, n * 8 // 2)):
    #     encoded_bytes = []
    #     for i in range(0, len(bs), 8):
    #         encoded_bytes.append(int(bs[i:i + 8], base=2))
    #     if rs.check_fast(encoded_bytes):
    #         count += 1
    #         print(encoded_bytes)
    #     if count >= 10:
    #         print("********************* DONE *************")
    #         print(f"Did {loop+1} iterations")
    #         break
    #

    alpha = list(range(1 << k))
    from coders.source.source_coder import DummySourceCoder
    r = ReedSolomonCoder("rsc", prob=False, n=n, factor=0)
    from coders.utils import all_bit_strings, pop_count, n_bits_with_k_set
    from collections import defaultdict, OrderedDict
    from unireedsolomon.ff import find_prime_polynomials

    print(find_prime_polynomials(generator=3, c_exp=2))

    r.set_source_coder(DummySourceCoder("dummy", codes=list(all_bit_strings(k))))
    r.set_alphabet(alpha)
    dd = defaultdict(int)
    nonzero = defaultdict(int)
    r.set_alphabet(alpha)
    for sym in alpha:
        # print(sym)
        enc = r.encode(sym)
        p = pop_count(enc)
        encoded_bytes = [int(enc[i:i + 8], base=2) for i in range(0, len(enc), 8)]
        nonzero[sum([1 if b != 0 else 0 for b in encoded_bytes])] += 1
        dd[p] += 1

    print(OrderedDict(sorted(dd.items(), key=lambda t: t[0])))
    for c in range(n * 8 // 2 - n // 2, n * 8 // 2 + n // 2):
        if c == n * 8 // 2:
            print("********************************")
        if dd[c]:
            print(c, dd[c])
        if c == n * 8 // 2:
            print("********************************")

    print(OrderedDict(sorted(nonzero.items(), key=lambda t: t[0])))
    for c in range(n):
        if c == n // 2:
            print("********************************")
        if nonzero[c]:
            print(c, nonzero[c])
        if c == n // 2:
            print("********************************")
