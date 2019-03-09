from coders.channel.channel_coder import ChannelCoder
from utils.utils import encode_probs


class RepetitionCoder(ChannelCoder):
    """
    Encodes binary strings by repeating each bit/block multiple times to correct e errors.
    """
    def __init__(self, source_coder, repetition, method="bit"):
        super(RepetitionCoder, self).__init__(source_coder)
        self.d = repetition  # the hamming distance of this code
        self.max_correct = self.d // 2
        self.method = method

    @staticmethod
    def _majority_decode_bit(s):
        one_count = sum([1 if bit == "1" else 0 for bit in s])
        zero_count = sum([1 if bit == "0" else 0 for bit in s])
        assert one_count + zero_count == len(s)
        if one_count > zero_count:
            return "1"
        elif zero_count > one_count:
            return "0"
        else:
            raise Exception("This should not happen")

    @staticmethod
    def _majority_decode_block(blocks):
        return max(set(blocks), key=blocks.count)

    def encode(self, label, prob=False):
        bs = self.source_coder.encode(label)
        if self.method == "bit":
            encoding = "".join([str(bit) * self.d for bit in bs])
        elif self.method == "block":
            encoding = bs * self.d
        else:
            raise NotImplementedError

        if prob:
            encoding = encode_probs(encoding, self.max_correct, 0)
        return encoding

    def decode(self, bs):
        if self.method == "bit":
            decoding = ""
            for i in range(0, len(bs), self.d):
                decoding += RepetitionCoder._majority_decode_bit(bs[i:i+self.d])
        elif self.method == "block":
            blocks = []
            for i in range(0, len(bs), self.source_coder.output_size):
                blocks.append(bs[i:i + self.source_coder.output_size])
            decoding = RepetitionCoder._majority_decode_block(blocks)
        else:
            raise NotImplementedError
        return self.source_coder.decode(decoding)


if __name__ == "__main__":
    alpha = list(range(10))
    from coders.source.bcd_coder import BcdCoder
    r = RepetitionCoder(BcdCoder(alpha, allow_zero=False), repetition=5, method="block")
    for sym in alpha:
        print(sym)
        enc = r.encode(sym, True)
        print(enc)
        dec = r.decode(enc)
        print(dec)
        print("--")
