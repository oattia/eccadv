import logging

from coders.utils import bit_list_to_str

logger = logging.getLogger(__name__)


class SourceCoder:
    """
    In coding theory, source coder is an encoder/decoder object that has a predetermined alphabet.
    The source coder controls how every symbol in the alphabet should be encoded to be ready for the channel coder.
    """
    def __init__(self, name, allow_zero, output_size):
        self.name = name
        self.allow_zero = allow_zero
        self.output_size = output_size  # k, in bits

        # dataset dependant state
        self.alphabet = None
        self.symbol2code = None
        self.code2symbol = None

    def _build_code(self):
        """
        Fill the code/symbols dictionaries.
        """
        raise NotImplementedError

    def has_alphabet(self):
        """
        Returns True if the state is set, False otherwise.
        """
        return self.alphabet is not None

    def set_alphabet(self, alphabet):
        """
        Sets the dataset dependant state of the coder.
        """
        self.alphabet = [str(symbol) for symbol in sorted(alphabet)]
        self.symbol2code = {}
        self.code2symbol = {}
        self._build_code()

    def encode(self, symbol):
        """
        Returns an encoding of this symbol or fails.
        """
        assert self.has_alphabet()
        return self.symbol2code[str(symbol)]

    def decode(self, code):
        """
        Returns a decoding of this code or fails.
        """
        assert self.has_alphabet()
        return self.code2symbol[bit_list_to_str(code)]

    def has_code(self, code):
        """
        Returns True if code is included in the alphabet code, False otherwise
        """
        assert self.has_alphabet()
        return code in self.code2symbol


class DummySourceCoder(SourceCoder):
    """
    Dummy source coder, returns codes that are given to it.
    """
    def __init__(self, name, codes):
        super(DummySourceCoder, self).__init__(name, True, len(codes[0]))
        self.codes = codes

    def _build_code(self):
        assert len(self.alphabet) == len(self.codes)
        for symbol, code in zip(self.alphabet, self.codes[: len(self.alphabet)]):
            self.symbol2code[symbol] = code
            self.code2symbol[code] = symbol
