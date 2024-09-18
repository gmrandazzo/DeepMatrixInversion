#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

import argparse
from deepmatrixinversion.nnmodel import NN
from sys import argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputmx', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--exptarget', default=None,
                        type=str, help='Experimental target data matrix')
    parser.add_argument('--model', default=None, type=str,
                        help='Model directory')
    parser.add_argument('--inverseout', default="Model", type=str,
                        help='inverse output matrix file')

    args = parser.parse_args(argv[1:])
    if args.inputmx is None or args.model is None:
        print("ERROR! Please specify input file matrix and a model directory!")
        print("\n Prediction usage: %s --inputmx [input file] --inverseout [output file] --model [path of the model]" % (argv[0]))
        print("\n Prediction with target evaluation usage: %s --inputmx [input file] --exptarget [experimental target file] --inverseout [output file] --model [path of the model]" % (argv[0]))
    else:
        nn = NN(args.inputmx, args.exptarget)
        nn.predict(args.model, args.inverseout)


if __name__ == '__main__':
    main()
