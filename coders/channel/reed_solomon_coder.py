import math

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
        self.bit_pad = None
        self.max_correct = None
        self.rs = None

    def set_source_coder(self, source_coder):
        super(ReedSolomonCoder, self).set_source_coder(source_coder)
        self.k = math.ceil(self.source_coder.output_size / 8)
        self.bit_pad = 0 if self.source_coder.output_size % 8 == 0 else 8 - self.source_coder.output_size % 8
        self.max_correct = (self.n - self.k) // 2
        assert self.max_correct >= 1
        self.rs = urs.RSCoder(self.n, self.k)

    def encode(self, label):
        assert self.is_set()
        bs = ("0" * self.bit_pad) + self.source_coder.encode(label)
        encoded_bytes = [int(x) for x in self.rs.encode([int(bs[i:i + 8], base=2) for i in range(0, len(bs), 8)], return_string=False)]
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
            decoding = "".join([int_to_bit_str(n, length=8) for n in decoded_bytes])
            return self.source_coder.decode(decoding[self.bit_pad:])
        except Exception as e:
            return "{} because {}".format(ChannelCoder.CANT_DECODE, str(e))

    def output_size(self):
        return self.n * 8


if __name__ == "__main__":
    alpha = list(range(8))
    from coders.source.bcd_coder import BcdCoder
    s = BcdCoder(name="bcd", allow_zero=True, output_size=-1)
    s.set_alphabet(alpha)
    print(s.code2symbol)

    r = ReedSolomonCoder("rs", prob=True, factor=-100, n=8)
    r.set_source_coder(s)

    for sym in alpha:
        print(sym)
        enc = r.encode(sym)
        print(enc)
        dec = r.decode(enc)
        print(dec)
        print("--")


if __name__ == "__main2__":

    from coders.source.source_coder import DummySourceCoder
    r = ReedSolomonCoder("rsc", prob=True, n=8, factor=1000)
    from coders.utils import all_bit_strings, pop_count, n_bits_with_k_set
    from collections import defaultdict, OrderedDict
    count = 0
    for loop, bs in enumerate(n_bits_with_k_set(n * 8, n * 8 // 2)):
        encoded_bytes = []
        for i in range(0, len(bs), 8):
            encoded_bytes.append(int(bs[i:i + 8], base=2))
        if rs.check_fast(encoded_bytes):
            count += 1
            print(encoded_bytes)
        if count >= 10:
            print("********************* DONE *************")
            print(f"Did {loop+1} iterations")
            break


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
