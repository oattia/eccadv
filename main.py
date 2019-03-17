import argparse
import logging 

from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()    
    config = Config(args.config)
    for ex_id, ex in tqdm(config.experiments.items(), desc="Executing experiments"):
        logger.info("Starting experiment {}".format(ex_id))
        ex.run()
        logger.info("Finished experiment {}".format(ex_id))
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
