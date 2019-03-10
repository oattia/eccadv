from coders.channel.channel_coder import ChannelCoder


class HammingCoder(ChannelCoder):
    def __init__(self,  name, prob, factor):
        super(HammingCoder, self).__init__( name, prob, factor)

    def encode(self, bs, prob=False):
        pass

    def decode(self, bs):
        pass

    def output_size(self):
        pass
