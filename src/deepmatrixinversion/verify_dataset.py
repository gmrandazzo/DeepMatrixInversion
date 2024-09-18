import sys

from deepmatrixinversion.dataset import verify_matrix_inversion, verify_pseudo_inversion
from deepmatrixinversion.io import read_dataset


def main():
    if len(sys.argv) != 5:
        print(
            f"Usage {sys.argv[0]} [dataset matrix to invert] [dataset matrix inverted] [dataset singular matrix to invert] [dataset singular matrix inverted]"
        )
        exit(0)
    X = read_dataset(sys.argv[1])
    Y = read_dataset(sys.argv[2])
    X_Singular = read_dataset(sys.argv[1])
    Y_Singular = read_dataset(sys.argv[2])

    if verify_matrix_inversion(X, Y) and verify_pseudo_inversion(
        X_Singular, Y_Singular
    ):
        print("Dataset valid.")
    else:
        print("WARNING: Dataset not valid!")


if __name__ in "__main__":
    main()
