from coders.channel.channel_coder import ChannelCoder
from utils.utils import encode_probs


class RepetitionCoder(ChannelCoder):
    """
    Encodes binary strings by repeating each bit/block multiple times to correct e errors.
    """
    def __init__(self, name, prob, factor, repetition, method):
        super(RepetitionCoder, self).__init__(name, prob, factor)
        self.d = repetition  # the hamming distance of this code
        self.method = method
        self.max_correct = None

    def set_source_coder(self, source_coder):
        super(RepetitionCoder, self).set_source_coder(source_coder)
        self.max_correct = self.d // 2
        assert self.max_correct >= 1

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
            raise Exception("This should never happen!")

    @staticmethod
    def _majority_decode_block(blocks):
        return max(set(blocks), key=blocks.count)

    def encode(self, label):
        assert self.is_set()
        bs = self.source_coder.encode(label)
        if self.method == "bit":
            encoding = "".join([str(bit) * self.d for bit in bs])
        elif self.method == "block":
            encoding = bs * self.d
        else:
            raise Exception(f"Invalid encoding method {self.method} for {self.name}")

        if self.prob:
            encoding = encode_probs(encoding, self.max_correct, self.factor)
        return encoding

    def decode(self, bs):
        try:
            assert self.is_set()
            bs = "".join([str(x) for x in bs])
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
                raise Exception(f"Invalid decoding method {self.method} for {self.name}")
            return self.source_coder.decode(decoding)
        except Exception as e:
            return f"{ChannelCoder.CANT_DECODE} because {e}"

    def output_size(self):
        return self.d * self.source_coder.output_size


if __name__ == "__main__":
    alpha = list(range(10))
    from coders.source.bcd_coder import BcdCoder
    r = RepetitionCoder("rc", prob=True, factor=1, repetition=5, method="block")
    r.set_source_coder(BcdCoder(name="bcd", allow_zero=False, output_size=-1))
    r.set_alphabet(alpha)
    for sym in alpha:
        print(sym)
        enc = r.encode(sym)
        print(enc)
        dec = r.decode(enc)
        print(dec)
        print("--")
