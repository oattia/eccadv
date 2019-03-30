import logging

import numpy as np
from scipy.special import comb

from coders.source.source_coder import SourceCoder
from coders.utils import n_bits_with_k_set

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WeightCoder(SourceCoder):
    """
    Outputs an encoding for each symbol that has a fixed weight.
    """
    def __init__(self, name, output_size, weight):
        super(WeightCoder, self).__init__(name, False, output_size)
        self.weight = weight

    def _build_code(self):
        min_output_size = int(np.ceil(np.log2(len(self.alphabet))))

        if self.output_size < min_output_size:
            self.output_size = min_output_size

        # look for the lowest output_size that can encode all symbols
        while comb(self.output_size, self.weight, exact=True) < len(self.alphabet):
            self.output_size += 1

        i = 0
        for encoding in n_bits_with_k_set(self.output_size, self.weight):
            if i == len(self.alphabet):
                break
            self.symbol2code[self.alphabet[i]] = encoding
            self.code2symbol[encoding] = self.alphabet[i]
            i += 1


if __name__ == "__main__":
    alpha = list(range(16))
    b = WeightCoder("wc", 6, 3)
    b.set_alphabet(alpha)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
