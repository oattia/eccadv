import logging

import numpy as np

from scoder.source_coder import SourceCoder
from utils.utils import int_to_bit_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BcdCoder(SourceCoder):
    """
    Outputs an encoding for each symbol that is equal to its index's BCD representation, padded to equal length.
    """
    def __init__(self, alphabet, allow_zero):
        super(BcdCoder, self).__init__(alphabet, allow_zero)

    def _build_code(self):
        self.output_size = int(np.ceil(np.log2(len(self.alphabet))))
        if (not self.allow_zero) and ((1 << self.output_size) == len(self.alphabet)):
            self.output_size += 1
        for i, symbol in enumerate(self.alphabet, start=(0 if self.allow_zero else 1)):
            encoding = int_to_bit_str(i, self.output_size)
            self.symbol2code[symbol] = encoding
            self.code2symbol[encoding] = symbol


if __name__ == "__main__":
    alpha = list(range(16))
    b = BcdCoder(alpha, True)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
