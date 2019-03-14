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
        self.rs: urs.RSCoder = None

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
            encoded_bytes = []
            for i in range(0, len(bs), 8):
                encoded_bytes.append(int(bs[i:i + 8], base=2))
            decoded_bytes, _ = self.rs.decode(encoded_bytes, return_string=False, nostrip=True)
            decoding = "".join([str(bit) for bit in decoded_bytes])
            return self.source_coder.decode(decoding)
        except Exception as e:
            return f"{ChannelCoder.CANT_DECODE} because {e}"

    def output_size(self):
        return self.n * 8


if __name__ == "__main__":
    alpha = list(range(10))
    from coders.source.rand_coder import RandomCoder
    r = ReedSolomonCoder("rsc", prob=True, n=10)
    r.set_source_coder(RandomCoder("rc", allow_zero=False, output_size=-1))
    r.set_alphabet(alpha)
    for sym in alpha:
        print(sym)
        enc = r.encode(sym)
        dec = r.decode(enc)
        print(dec)
        print("--")
