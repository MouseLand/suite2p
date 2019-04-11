import suite2p
import numpy as np
import time, os
from suite2p import register, dcnv, celldetect, run_s2p, gui2p
import argparse

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def main():
    ops = np.load('ops.npy')
    ops = ops.item()
    suite2p.main(ops)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Suite2p parameters')
    parser.add_argument('--ops', default=[], type=str, help='options')
    parser.add_argument('--db', default=[], type=str, help='options')
    args = parser.parse_args()

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    ops = {}
    db= {}
    if len(args.ops)>0:
        ops = np.load(args.ops)
        ops = ops.item()
    if len(args.db)>0:
        db = np.load(args.db)
        db = db.item()
        suite2p.run_s2p.run_s2p(ops, db)
    else:
        gui2p.run()
