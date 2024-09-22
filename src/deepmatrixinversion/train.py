#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

import argparse
from sys import argv

from deepmatrixinversion.nnmodel import NN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msize", default=None, type=str, help="matrix size")
    parser.add_argument("--nmx_samples", default=1000000, type=int, help="matrix size")
    parser.add_argument(
        "--nunits", default=128, type=int, help="number of units resnet"
    )
    parser.add_argument(
        "--nlayers", default=7, type=int, help="number of resnet layers"
    )
    parser.add_argument("--rmin", default=-1, type=float, help="min range matrix")
    parser.add_argument("--rmax", default=1, type=float, help="max range matrix")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size")
    parser.add_argument("--n_repeats", default=20, type=int, help="Number of models")
    parser.add_argument("--mout", default="Model", type=str, help="Model out path")
    parser.add_argument(
        "--plotout",
        default=None,
        type=str,
        help="Plot validation output to screen or file",
    )

    args = parser.parse_args(argv[1:])
    if args.msize is None:
        print("ERROR! Please specify matrix size you want to have!")
        print(
            "\n Usage: %s --msize [input file] --rmin [float] --rmax [float] --epochs [int] --batch_size [int] --mout [str] --n_repeats [int]"
            % (argv[0])
        )
    else:
        nn = NN(args.msize, args.rmin, args.rmax)
        nn.verbose = 1
        nn.train(
            args.nmx_samples,
            args.batch_size,
            args.epochs,
            args.n_repeats,
            args.mout,
        )

        nn.model_validator(
            mout_path=args.mout, nmx_sample=args.nmx_sample, plotout=args.plotout
        )


if __name__ == "__main__":
    main()
