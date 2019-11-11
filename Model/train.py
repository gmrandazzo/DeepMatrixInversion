#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2019 gmrandazzo@gmail.com
# This file is part of DeepMatrixInversion.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
from nnmodel import NN
from sys import argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputmx', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--inversemx', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Batch size')
    parser.add_argument('--n_splits', default=5, type=int,
                        help='Number of kfold splits')
    parser.add_argument('--n_repeats', default=20, type=int,
                        help='Number of repetitions')
    parser.add_argument('--nunits', default=32, type=int,
                        help='Number of neurons')
    parser.add_argument('--mout', default="Model", type=str,
                        help='Model out path')

    args = parser.parse_args(argv[1:])
    if args.inputmx is None or args.inversemx is None:
        print("ERROR! Please specify input table to train!")
        print("\n Usage: %s --inputmx [input file] --inversemx [input file] --epochs [int] --batch_size [int] --nunits [int]" % (argv[0]))
    else:
        nn = NN(args.inputmx, args.inversemx)
        nn.train(args.batch_size,
                 args.epochs,
                 args.nunits,
                 args.n_splits,
                 args.n_repeats,
                 args.mout)


if __name__ == '__main__':
    main()
