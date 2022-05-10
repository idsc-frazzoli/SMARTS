import argparse
import subprocess
import os
from pathlib import Path
import pandas as pd


def open_image(path):
    subprocess.run(["xdg-open", path])


def get_convergence_log_paths(path):
    convergence_log_paths = []
    all_paths = list(os.walk(Path(path), topdown=True))
    for x in all_paths:
        if "convergence_logs.csv" in x[2] and "logs_and_plots" not in x[0]:
            convergence_log_paths.append(x[0])

    return list(set(convergence_log_paths))


def main(path):
    convergence_log_paths = get_convergence_log_paths(path)

    print(convergence_log_paths)

    for conv_path in convergence_log_paths:

        manual_log_keys = {"date", "time", "run", "paradigm", "name", "converged", "stable_checkpoint", "first_stable",
                           "total_checkpoints", "converged_len", "max_reward", "max_reward_checkpoint", "run_path",
                           "human_converged"}
        manual_logs = dict([(key, []) for key in manual_log_keys])

        df_logs = pd.read_csv(Path(conv_path, "convergence_logs.csv"))
        for index, row in df_logs.iterrows():
            for key in manual_log_keys:
                if key != "human_converged":
                    manual_logs[key].append(row[key])

            if not row["converged"]:
                image_path = Path(conv_path + "/convergence_analysis_" + row["name"] + ".png")
                print(image_path)
                open_image(image_path)

                human_input = False
                while human_input not in ("0", "1"):
                    human_input = input()

                manual_logs["human_converged"].append(int(human_input))
            else:
                manual_logs["human_converged"].append(1)

        df_logs = pd.DataFrame.from_dict(manual_logs)
        df_logs.to_csv(Path(conv_path, "convergence_logs_manual.csv"))


def parse_args():
    parser = argparse.ArgumentParser("")

    parser.add_argument("-p",
                        "--path",
                        type=str,
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(path=args.path)
