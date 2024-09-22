import sys

from deepmatrixinversion.dataset import verify_matrix_inversion, verify_pseudo_inversion
from deepmatrixinversion.io import read_dataset


def main():
    if len(sys.argv) != 4:
        print(
            f"Usage {sys.argv[0]} [dataset matrix to invert] [dataset matrix inverted] [type: invertible or singular]"
        )
        exit(0)
    X = read_dataset(sys.argv[1])
    Y = read_dataset(sys.argv[2])

    if sys.argv[3] == "invertible":
        verify = verify_matrix_inversion
    else:
        verify = verify_pseudo_inversion

    if verify(X, Y):
        print("Dataset valid.")
    else:
        print("WARNING: Dataset not valid!")


if __name__ in "__main__":
    main()
