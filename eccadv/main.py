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
    for ex_id, ex in config.experiments.items():
        logger.info("Starting experiment {}".format(ex_id))
        summary[ex_id] = ex.run()
        logger.info("Finished experiment {}".format(ex_id))
        logger.info("=" * 40)
        logger.info("=" * 40)

    for ex_id, result in summary.items():
        print(ex_id + ": " + str(result))


if __name__ == "__main__":
    main()
