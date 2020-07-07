import argparse
import suite2p
import numpy as np
import matplotlib.pyplot as plt

from suite2p.__main__ import add_args, parse_args


def set_op_param(ops, args, param_name_list):
    for p_name in param_name_list:
        pval = getattr(args, p_name)
        if pval:
            ops[p_name] = pval
            print('->> Setting {0} to {1}'.format(p_name, ops[p_name]))
    return ops


def registration_metrics():
    default_parser = add_args(argparse.ArgumentParser(description='Suite2p parameters'))
    default_parser.add_argument('data_path', type=str, nargs=1, help='Path to directory with input files')
    default_parser.add_argument('--tiff_list', default=[], type=str, nargs='*', help='Input files selected')
    args, ops = parse_args(default_parser)
    # Set registration metrics specific parameters
    ops['do_regmetrics'] = True
    ops['roidetect'] = False
    ops['reg_metric_n_pc'] = 10
    ops = set_op_param(ops, args, ['data_path', 'tiff_list'])
    print("Calculating registration metrics...")
    result_ops = suite2p.run_s2p(ops)
    for p in range(ops['nplanes']):
        dx = result_ops[p]['regDX']
        reg_pc = result_ops[p]['regPC']
        tpc = result_ops[p]['tPC']
        avg_scores = np.mean(tpc, axis=1)
        max_scores = np.max(tpc, axis=1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].plot(avg_scores)
        ax[0].title.set_text('Avg scores')
        ax[1].plot(max_scores)
        ax[1].title.set_text('Max scores')
        plt.show()


if __name__ == "__main__":
    registration_metrics()