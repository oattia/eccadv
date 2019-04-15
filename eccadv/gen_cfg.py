
pattern = """
  exp_{}:
    seed: 1234
    dataset: "mnist"
    source_coder: "{}"
    channel_coder: "{}"
    model: "{}"
    attacker: "{}"
    thresholding:
      method: "max_k"
      k: {}
"""

from collections import OrderedDict

def main():
    models = ["ch_cnn_ce",
              "ch_cnn_mae",
              "ch_cnn_be"]

    attackers = [
                # "dummy",
                 # "fgsm_ch_0.05", "fgsm_ch_0.1", "fgsm_ch_0.2",
                 "fgsm_ch_0.3"]

    pro_methods = OrderedDict([
        # ("oh", [
        #         # ("dummy", 1),
        #         ("rep_11_block", 11),
        #         # ("rep_11_bit", 11),
        #         # ("rep_5_block", 5), ("rep_5_bit", 5)
        #         ]),
        # ("weight_3_6", [("rep_11_block", 33),
        #                 # ("rep_11_bit", 33),
        #                 # ("rep_5_block", 15), ("rep_5_bit", 15)
        #                 ]),
        # # ("rand_5", [("hd", 16)]),
        # ("rand_10", [("hd", 512)]),
        # # ("dummy_rs_3_12", [("dummy", 12)]),
        # ("dummy_rs_4_16", [("dummy", 16)]),
        # # ("dummy_rs_4_6", [("dummy", 6)])
        ("dummy_rs_128_512", [("dummy", 512)])
    ])

    exp = 800
    for model in models:
        for attacker in attackers:
            for pro_method in pro_methods.items():
                scoder, l_coder = pro_method
                for ccoder, thres in l_coder:
                    xx = pattern.format(str(exp),
                               scoder,
                               ccoder,
                               model,
                               attacker,
                               thres)
                    exp += 1
                    print(xx)
    print(exp)


if __name__ == "__main__":
    main()
