class ChannelCoder:
    """
    In coding theory, channel coder is an encoder/decoder object that maps binary strings to valid codewords by adding
     redundancy that is able to correct some bit errors depending on the coding scheme itself.
    """

    CANT_DECODE = "CANT_DECODE"

    def __init__(self, name, prob, factor):
        self.name = name
        self.prob = prob
        self.factor = factor
        self.source_coder = None
    
    def encode(self, s):
        raise NotImplementedError
    
    def decode(self, s):
        raise NotImplementedError

    def output_size(self):
        raise NotImplementedError

    def is_set(self):
        return self.source_coder is not None and self.source_coder.has_alphabet()

    def set_source_coder(self, source_coder):
        self.source_coder = source_coder


class DummyChannelCoder(ChannelCoder):
    """
    Dummy Coder that does no error correction, used for comparison.
    """
    def __init__(self, name, prob=False, factor=None):
        super(DummyChannelCoder, self).__init__(name, False, None)

    def encode(self, s):
        assert self.is_set()
        return self.source_coder.encode(s)

    def decode(self, s):
        assert self.is_set()
        return self.source_coder.decode(s)

    def output_size(self):
        assert self.is_set()
        return self.source_coder.output_size
