
import argparse
from functools import lru_cache


@lru_cache
def fibonacci_of(n):
    if n in {0, 1}:  # Base case
        return n
    return fibonacci_of(n - 1) + fibonacci_of(n - 2)  # Recursive case


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n",
        type=int,
        default=5,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(fibonacci_of(n=args.n))