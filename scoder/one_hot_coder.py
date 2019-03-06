import logging

from scoder.source_coder import SourceCoder
from utils.utils import bit_list_to_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OneHotCoder(SourceCoder):
    """
    Outputs an encoding for each symbol in which 1 is placed for its index and the rest is set to 0.
    """
    def __init__(self, alphabet):
        super(OneHotCoder, self).__init__(alphabet, False)

    def _build_code(self):
        self.output_size = len(self.alphabet)
        for symbol in self.alphabet:
            encoding = bit_list_to_str([1 if symbol == s else 0 for s in self.alphabet])
            self.symbol2code[symbol] = encoding
            self.code2symbol[encoding] = symbol


if __name__ == "__main__":
    alpha = list(range(10))
    b = OneHotCoder(alpha)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
