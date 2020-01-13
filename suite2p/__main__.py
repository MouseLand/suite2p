import numpy as np
import os
import argparse

def main():
    ops = np.load('ops.npy', allow_pickle=True).item()
    import suite2p
    suite2p.main(ops)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Suite2p parameters')
    parser.add_argument('--ops', default=[], type=str, help='options')
    parser.add_argument('--db', default=[], type=str, help='options')
    args = parser.parse_args()

    ops = {}
    db= {}
    if len(args.ops)>0:
        ops = np.load(args.ops, allow_pickle=True).item()
    if len(args.db)>0:
        db = np.load(args.db, allow_pickle=True).item()
        from . import run_s2p
        run_s2p.run_s2p(ops, db)
    else:
        from .gui import gui2p
        gui2p.run()

    
if __name__ == '__main__':
    parse_arguments()
