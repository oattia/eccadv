import logging

import numpy as np

from coders.source.source_coder import SourceCoder
from utils.utils import gray_code, int_to_bit_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GrayCoder(SourceCoder):
    """
    Outputs an encoding for each symbol that is equal to its index's gray binary code, padded to equal length.
    """
    def __init__(self, name, allow_zero, output_size):
        super(GrayCoder, self).__init__(name, allow_zero, output_size)

    def _build_code(self):
        min_output_size = int(np.ceil(np.log2(len(self.alphabet))))
        if (not self.allow_zero) and ((1 << min_output_size) == len(self.alphabet)):
            min_output_size += 1
        if self.output_size < min_output_size:
            self.output_size = min_output_size
        for i, symbol in enumerate(self.alphabet, start=(0 if self.allow_zero else 1)):
            encoding = int_to_bit_str(gray_code(i), self.output_size)
            self.symbol2code[symbol] = encoding
            self.code2symbol[encoding] = symbol


if __name__ == "__main__":
    alpha = list(range(16))
    b = GrayCoder("gc", False, -1)
    b.set_alphabet(alpha)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
