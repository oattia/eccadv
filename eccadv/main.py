import argparse
import logging
import sys

from config import Config


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()    
    config = Config(args.config)
    summary = {}
    results_latex = open("results_latex.txt", "w")
    results_table = open("results_table.txt", "w")
    sep = "--" * 25
    for ex_id, ex in config.experiments.items():
        logger.info("Starting experiment {}".format(ex_id))
        try:
            result = ex.run()
            summary[ex_id] = result
            logger.info("Finished experiment {}".format(ex_id))
            logger.info(sep)
            res_str = result.to_string(index=False)

            print(ex_id)
            print(sep)
            print(res_str)

            results_latex.write("{}\n{}\n{}\n{}\n".format(ex_id, sep, result.to_latex(index=False), sep))
            results_table.write("{}\n{}\n{}\n{}\n".format(ex_id, sep, res_str, sep))
        except:
            logger.info("experiment {} Failed because of {}".format(ex_id, str(sys.exc_info()[0])))
            logger.info(sep)

    results_latex.close()
    results_table.close()


if __name__ == "__main__":
    main()
