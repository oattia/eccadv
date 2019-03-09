from coders.source.source_coder import  SourceCoder


class ChannelCoder:
    """
    In coding theory, channel coder is an encoder/decoder object that maps binary strings to valid codewords by adding
        and redundancy is able to correct some bit errors depending on the code itself.
    """
    def __init__(self, source_coder: SourceCoder):
        self.source_coder = source_coder
    
    def encode(self, s, prob=False):
        raise NotImplementedError
    
    def decode(self, s):
        raise NotImplementedError
