from coders.channel.channel_coder import ChannelCoder


class TruboCoder(ChannelCoder):
    def __init__(self, name, prob, factor):
        super(TruboCoder, self).__init__(name, prob, factor)

    def encode(self, s, prob=False):
        pass

    def decode(self, s):
        pass

    def output_size(self):
        pass
