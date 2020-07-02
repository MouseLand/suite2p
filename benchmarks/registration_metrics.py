import argparse
import suite2p
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
    ops = set_op_param(ops, args, ['data_path', 'tiff_list'])
    print("Calculating registration metrics...")
    result_ops = suite2p.run_s2p(ops)
    for p in range(ops['nplanes']):
        dx = result_ops[p]['regDX']
        reg_pc = result_ops[p]['regPC']
        tpc = result_ops[p]['tPC']
        bottom_pc = reg_pc[0]
        top_pc = reg_pc[1]
        print(top_pc.shape)
        print(bottom_pc.shape)


if __name__ == "__main__":
    registration_metrics()