import logging

from coders.source.source_coder import SourceCoder
from coders.utils import bit_list_to_str

logger = logging.getLogger(__name__)


class OneHotCoder(SourceCoder):
    """
    Outputs an encoding for each symbol in which 1 is placed for its index and the rest is set to 0.
    """
    def __init__(self, name, output_size, allow_zero=False):
        # ignore allow zeros
        super(OneHotCoder, self).__init__(name, False, output_size)

    def _build_code(self):
        min_output_size = len(self.alphabet)
        if self.output_size < min_output_size:
            self.output_size = min_output_size
        for symbol in self.alphabet:
            encoding = (self.output_size - len(self.alphabet)) * "0" + \
                       bit_list_to_str([1 if symbol == s else 0 for s in self.alphabet])
            self.symbol2code[symbol] = encoding
            self.code2symbol[encoding] = symbol


if __name__ == "__main__":
    alpha = list(range(10))
    b = OneHotCoder("ohc", output_size=-1)
    b.set_alphabet(alpha)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
