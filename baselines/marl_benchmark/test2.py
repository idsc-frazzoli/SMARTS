import argparse


def main(x):
    print("the number is {}".format(x))


def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument("--x",
                        type=str,
                        default="",
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(x=args.x)
