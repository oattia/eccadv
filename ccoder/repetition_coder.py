
from ccoder.channel_coder import ChannelCoder


class RepetitionCoder(ChannelCoder):
    def __init__(self, repetition):
        super(RepetitionCoder, self).__init__()
        self.repetition = repetition

    def encode(self, ds):
        pass

    def decode(self, ds):
        pass
