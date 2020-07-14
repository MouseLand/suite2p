import argparse
import suite2p
import numpy as np

from suite2p.__main__ import add_args, parse_args


def print_iter(iter_list, iter_name_list, pre=""):
    for val, name in zip(iter_list, iter_name_list):
        print("{} offsets: {}".format(pre + " " + name, val))


def registration_metrics():
    """
    Displays registration offsets calculated on pclow and pchigh frames. If registration was performed well,
    the PCs should not contain movement. All offsets calculated on pclow/pchigh frames should be close to zero.
    """
    default_parser = add_args(argparse.ArgumentParser(description='Suite2p parameters'))
    default_parser.add_argument('data_path', type=str, nargs=1, help='Path to directory with input files')
    default_parser.add_argument('--tiff_list', default=[], type=str, nargs='*', help='Input files selected')
    args, ops = parse_args(default_parser)
    # Set registration metrics specific parameters
    ops['do_regmetrics'] = True
    ops['roidetect'] = False
    ops['reg_metric_n_pc'] = 10
    # Sets each parameter name to corresponding args value for ops dictionary
    for p_name in ['data_path', 'tiff_list']:
        pval = getattr(args, p_name)
        if pval:
            ops[p_name] = pval
            print('->> Setting {0} to {1}'.format(p_name, ops[p_name]))
    print("Calculating registration metrics...")
    result_ops = suite2p.run_s2p(ops)
    off_strs = ['rigid', 'avg_non_rigid', 'max_non_rigid']
    for p in range(ops['nplanes']):
        print("\nPlane {}: ".format(p))
        offsets = result_ops[p]['regDX']
        avg_offs = np.mean(offsets, axis=0)
        max_offs = np.max(offsets, axis=0)
        print_iter(avg_offs, off_strs, 'Average')
        print_iter(max_offs, off_strs, 'Max')


if __name__ == "__main__":
    registration_metrics()
