import numpy as np

from coders.channel.channel_coder import ChannelCoder
from coders.utils import encode_probs, all_bit_strings, int_to_bit_str


class HadamardCoder(ChannelCoder):
    """
    Encodes binary strings using Hadamard code.
    Resources:
    - http://homepages.math.uic.edu/~leon/mcs425-s08/handouts/Hadamard_codes.pdf
    - http://www.mcs.csueastbay.edu/~malek/TeX/Hadamard.pdf
    - https://web.stanford.edu/class/ee387/handouts/notes09.pdf
    - https://web.stanford.edu/class/ee387/handouts/notes10.pdf
    - https://mat-web.upc.edu/people/sebastia.xambo/TC08/TC08-6-HadamardCodes.pdf
    """
    def __init__(self, name, prob, factor):
        super(HadamardCoder, self).__init__(name, prob, factor)
        self.n = None
        self.k = None
        self.max_correct = None

        # code matrices
        self.G = None
        self.H = None
        self.R = None
        self.lookup_table = None

    def set_source_coder(self, source_coder):
        super(HadamardCoder, self).set_source_coder(source_coder)
        self.k = self.source_coder.output_size
        self.n = (1 << self.k)
        self.max_correct = ((self.n // 2) - 1) // 2
        assert self.max_correct >= 1
        self.G, self.H, self.R = HadamardCoder._build_code(self.k)
        self.lookup_table = HadamardCoder._hadamard_matrix(self.n)

    @staticmethod
    def _build_code(k):
        all_binary_strings = list(all_bit_strings(k))
        # make all_binary_strings systematic by putting I_k in the end
        for i in reversed(range(k)):
            all_binary_strings.append(all_binary_strings.pop(1 << i))

        # map to save the transformation in columns
        # the indices must be stored to generate a lookup table that is in the same order as G and R
        idx_mapper = {}
        for i in range(1 << k):
            idx_mapper[int(all_binary_strings[i], base=2)] = i

        idx_formatter = [idx_mapper[i] for i in range(1 << k)]

        # systematic generator matrix ( k x n )
        # In coding theory, a generator matrix is a matrix whose rows form a basis for a linear code.
        # The codewords are all of the linear combinations of the rows of this matrix, that is,
        # the linear code is the row space of its generator matrix.
        G = np.zeros((k, 1 << k), dtype=int)
        for i in range(1 << k):
            G[:, i] = [int(bit) for bit in all_binary_strings[i]]

        # (k x (n-k) )
        P = G[:, :-k]

        # systematic parity check matrix ( (n-k) x n )
        # In coding theory, a parity-check matrix of a linear block code C is a matrix which describes
        # the linear relations that the components of a codeword must satisfy.
        H = np.hstack([np.eye((1 << k) - k, dtype=int), P.T])

        # k x n --- n x (n-k)
        assert ((G.dot(H.T) % 2) == np.zeros((k, (1 << k) - k), dtype=int)).all()

        # systematic pseudoinverse of G in the field ( n x k )
        R = np.vstack([np.zeros(((1 << k) - k, k), dtype=int), np.eye(k, dtype=int)])

        # Return the matrices in the right order for the lookup table
        return G[:, idx_formatter], H[:, idx_formatter], R[idx_formatter, :]

    @staticmethod
    def _hadamard_matrix(n):
        # assert n is a power of two and not zero
        assert (n != 0) and ((n & (n - 1)) == 0)
        # Initialize Hadamard matrix of order n.
        i1 = 1
        h_matrix = np.zeros((n, n), dtype=bool)
        while i1 < n:
            for i2 in range(i1):
                for i3 in range(i1):
                    h_matrix[i2 + i1][i3] = h_matrix[i2][i3]
                    h_matrix[i2][i3 + i1] = h_matrix[i2][i3]
                    h_matrix[i2 + i1][i3 + i1] = not h_matrix[i2][i3]
            i1 += i1

        h_matrix = h_matrix.astype(int)
        h_matrix[h_matrix == 0] = -1
        return h_matrix

    def encode(self, label):
        assert self.is_set()
        m = np.array([int(bit) for bit in self.source_coder.encode(label)], dtype=int)
        c = m.dot(self.G) % 2
        encoding = "".join([str(bit) for bit in c])
        if self.prob:
            encoding = encode_probs(encoding, self.max_correct, self.factor)
        return encoding

    def decode(self, bs):
        try:
            assert self.is_set()
            # received codeword
            r = np.array([int(x) for x in bs], dtype=int)
            r[r == 0] = -1

            # syndrome
            s = np.abs(r.dot(self.lookup_table))
            max_comp, max_pos = s.max(), s.argmax()
            if max_comp < self.n / 2:
                raise Exception("Too many errors!")

            # denoised codeword
            c = self.lookup_table[max_pos, :].copy()
            c[c == -1] = 0

            # actual decoding
            m = c.dot(self.R) % 2

            decoding = "".join([str(bit) for bit in m])
            return self.source_coder.decode(decoding)
        except Exception as e:
            return "{} because {}".format(ChannelCoder.CANT_DECODE, str(e))

    def output_size(self):
        return self.n


if __name__ == "__main__":
    alpha = list(range(20))
    from coders.source.bcd_coder import BcdCoder
    s = BcdCoder(name="bcd", allow_zero=True, output_size=-1)
    s.set_alphabet(alpha)
    print(s.code2symbol)

    r = HadamardCoder("rc", prob=False, factor=-100)
    r.set_source_coder(s)
    print(r.lookup_table)

    for sym in alpha:
        print(sym)
        enc = r.encode(sym)
        print(enc)
        dec = r.decode(enc)
        print(dec)
        print("--")
