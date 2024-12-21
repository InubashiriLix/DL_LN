import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_set",
                    default=3,
                    type=int,
                    required=True,
                    help="calculate the subset of a given set length ")
args = parser.parse_args()


def calculate(num: int):
    return num ** 2 if num > 0 else 0

print(calculate(args.n_set))
