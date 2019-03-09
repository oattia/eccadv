import logging

import numpy as np

from coders.source.source_coder import SourceCoder
from utils.utils import all_bit_strings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RandomCoder(SourceCoder):
    """
    Outputs an encoding for each symbol that is a random binary string.
    """
    def __init__(self, alphabet, allow_zero):
        super(RandomCoder, self).__init__(alphabet, allow_zero)

    def _build_code(self):
        self.output_size = int(np.ceil(np.log2(len(self.alphabet))))
        if (not self.allow_zero) and ((1 << self.output_size) == len(self.alphabet)):
            self.output_size += 1
        all_possible_encodings = all_bit_strings(self.output_size)
        if not self.allow_zero:
            all_possible_encodings.pop(0)
        encoding_subset = np.random.choice(all_possible_encodings, len(self.alphabet), replace=False)
        for symbol, encoding in zip(self.alphabet, encoding_subset):
            self.symbol2code[symbol] = encoding
            self.code2symbol[encoding] = symbol


if __name__ == "__main__":
    alpha = list(range(16))
    b = RandomCoder(alpha, False)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
