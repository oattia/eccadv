import argparse
import gc
import sys
import os

from config import Config


def get_run_exps():
    run = set()
    results_table = open("results_table.txt", "r")
    for line in results_table:
        if line.startswith("exp_"):
            run.add(line)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()    
    config = Config(args.config)
    summary = {}
    run_exps = get_run_exps()
    results_latex = open("results_latex.txt", "a")
    results_table = open("results_table.txt", "a")
    sep = "--" * 25
    for ex_id, ex in config.experiments.items():
        if ex_id in run_exps:
            print("Skipping experiment {}".format(ex_id))
            continue

        print("Starting experiment {}".format(ex_id))
        try:
            result = ex.run()
            summary[ex_id] = result
            print("Finished experiment {}".format(ex_id))
            print(sep)
            res_str = result.to_string(index=False)

            print(ex_id)
            print(sep)
            print(res_str)

            results_latex.write("{}\n{}\n{}\n{}\n".format(ex_id, sep, result.to_latex(index=False), sep))
            results_table.write("{}\n{}\n{}\n{}\n".format(ex_id, sep, res_str, sep))
            results_latex.flush()
            results_table.flush()
            os.fsync(results_latex.fileno())
            os.fsync(results_table.fileno())
        except:
            print("experiment {} Failed because of {}".format(ex_id, str(sys.exc_info()[0])))
            print(sep)
        finally:
            try:
                ex.cleanup()
                gc.collect()
            except:
                pass

    results_latex.close()
    results_table.close()


if __name__ == "__main__":
    main()
