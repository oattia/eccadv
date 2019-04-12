import logging

import numpy as np

from coders.source.source_coder import SourceCoder
from coders.utils import all_bit_strings

logger = logging.getLogger(__name__)


class RandomCoder(SourceCoder):
    """
    Outputs an encoding for each symbol that is a random binary string.
    """
    def __init__(self, name, allow_zero, output_size):
        super(RandomCoder, self).__init__(name, allow_zero, output_size)

    def _build_code(self):
        min_output_size = int(np.ceil(np.log2(len(self.alphabet))))
        if (not self.allow_zero) and ((1 << min_output_size) == len(self.alphabet)):
            min_output_size += 1
        if self.output_size < min_output_size:
            self.output_size = min_output_size
        all_possible_encodings = list(all_bit_strings(self.output_size))
        if not self.allow_zero:
            all_possible_encodings.pop(0)
        encoding_subset = np.random.choice(all_possible_encodings, len(self.alphabet), replace=False)
        for symbol, encoding in zip(self.alphabet, encoding_subset):
            self.symbol2code[symbol] = encoding
            self.code2symbol[encoding] = symbol


if __name__ == "__main__":
    alpha = list(range(16))
    b = RandomCoder("rc", False, -1)
    b.set_alphabet(alpha)
    for sym in alpha:
        print(sym)
        enc = b.encode(sym)
        print(enc)
        dec = b.decode(enc)
        print(dec)
        print("--")
