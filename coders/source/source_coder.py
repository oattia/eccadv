import logging

from utils.utils import bit_list_to_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SourceCoder:
    """
    In coding theory, source coder is an encoder/decoder object that has a predetermined alphabet.
    The source coder controls how every symbol in the alphabet should be encoded to be ready for the channel coder.
    """
    def __init__(self, alphabet, allow_zero):
        self.alphabet = [str(symbol) for symbol in sorted(alphabet)]
        self.allow_zero = allow_zero
        self.output_size = -1  # k
        self.symbol2code = {}
        self.code2symbol = {}
        self._build_code()

    def _build_code(self):
        """
        Fill the code/symbols dictionaries and output size.
        """
        raise NotImplementedError

    def encode(self, symbol):
        """
        Returns an encoding of this symbol or fails.
        """
        return self.symbol2code[str(symbol)]

    def decode(self, code):
        """
        Returns a decoding of this code or fails.
        """
        return self.code2symbol[bit_list_to_str(code)]

    def has_code(self, code):
        return code in self.code2symbol
