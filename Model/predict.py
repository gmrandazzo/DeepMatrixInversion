#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2019 gmrandazzo@gmail.com
# This file is part of DeepMatrixInversion.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sys import argv
from keras import backend as K
# Some memory clean-up

import argparse
import numpy as np
from keras.models import load_model
from pathlib import Path
from train import ReadMatrix

K.clear_session()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputmx', default=None,
                        type=str, help='input data matrix')
    parser.add_argument('--msize', default=None,
                        type=int, help='square matrix size')
    parser.add_argument('--model', default=None, type=str,
                        help='Model directory')
    parser.add_argument('--inverseout', default="Model", type=str,
                        help='inverse output matrix file')

    args = parser.parse_args(argv[1:])
    if args.inputmx is None or args.model is None:
        print("ERROR! Please specify input file matrix and a model directory!")
        print("\n Usage: %s --inputmx [input file] --inverseout [output file]" % (argv[0]))
    else:
        # Load input matrix to predicts
        X = ReadMatrix(args.inputmx)
        # Load models
        p = Path(args.model).glob('**/*.h5')
        files = [x for x in p if x.is_file()]
        models = []
        for file_ in files:
            # print("Load %s" % (file_))
            models.append(load_model(str(file_)))
        inverse = []
        for row in X:
            inv = []
            for model in models:
                inv.append(model.predict(np.array([row])))
            inv = np.array(inv)
            inverse.append(inv.mean(axis=0).tolist()[0])
        fo = open(args.inverseout, "w")
        if args.msize is not None:
            for inv in inverse:
                c = 0
                for i in range(args.msize):
                    for j in range(args.msize-1):
                        fo.write("%.4f," % (inv[c]))
                        c += 1
                    fo.write("%.4f\n" % (inv[c]))
                    c += 1
                fo.write("END\n")

        else:
            for inv in inverse:
                for i in range(len(inv)-1):
                    fo.write("%.4f," % (inv[i]))
                fo.write("%.4f\n" % (inv[-1]))
        fo.close()


if __name__ == '__main__':
    main()
