import argparse

import numpy as np


def main(ops1):
    # cleaning stuff
    print('cleaning!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Suite2p parameters')
    parser.add_argument('ops_file', type=str, help='ops file path')
    args = parser.parse_args()
    ops1 = np.load(args.ops_file)

    main(ops1)
