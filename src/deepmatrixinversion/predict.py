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
from sys import argv

from deepmatrixinversion.analytics import plot_exp_vs_pred
from deepmatrixinversion.inference import MatrixInversionInference
from deepmatrixinversion.io import read_dataset, write_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputmx", default=None, type=str, help="Matrices to invert ")
    parser.add_argument("--invtarget", default=None, type=str, help="Inverted matrices")
    parser.add_argument("--model", default=None, type=str, help="Model directory")
    parser.add_argument(
        "--invert_mode", default="nn", type=str, help="Method: nn or numpy"
    )
    parser.add_argument(
        "--inverseout", default="Model", type=str, help="inverse output matrix file"
    )
    parser.add_argument(
        "--plotout", default=None, type=str, help="Plot output screen or file"
    )

    args = parser.parse_args(argv[1:])
    if args.inputmx is None or args.model is None:
        print("ERROR! Please specify input file matrix and a model directory!")
        print(
            "\n Prediction usage: %s --inputmx [Matrices to invert] --inverseout [output ML inverted matrices] --model [path of the model]"
            % (argv[0])
        )
        print(
            "\n Prediction with target evaluation usage: %s --inputmx [Matrices to invert] --invtarget [Experimental inverted matrices] --inverseout [output ML inverted matrices] --model [path of the model]"
            % (argv[0])
        )
    else:
        mxinference = MatrixInversionInference(
            models_path=args.model, invert_mode=args.invert_mode
        )
        mxlst = read_dataset(args.inputmx)
        invpred = mxinference.invert(mxlst)
        write_dataset(invpred, args.inverseout)
        if args.plotout and args.invtarget:
            invmxlst = read_dataset(args.invtarget)
            plot_exp_vs_pred(
                invmxlst, invpred, None if args.plotout == "screen" else args.plotout
            )


if __name__ == "__main__":
    main()
