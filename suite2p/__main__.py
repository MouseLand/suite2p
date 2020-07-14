import argparse
import numpy as np
from suite2p import default_ops 


def add_args(parser: argparse.ArgumentParser):
    """
    Adds suite2p ops arguments to parser.
    """
    parser.add_argument('--ops', default=[], type=str, help='options')
    parser.add_argument('--db', default=[], type=str, help='options')
    ops0 = default_ops()
    for k in ops0.keys():
        v = dict(default=ops0[k], help='{0} : {1}'.format(k, ops0[k]))
        if k in ['fast_disk', 'save_folder', 'save_path0']:
            v['default'] = None
            v['type'] = str
        if (type(v['default']) in [np.ndarray, list]) and len(v['default']):
            v['nargs'] = '+'
            v['type'] = type(v['default'][0])
        parser.add_argument('--'+k, **v)
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = default_ops()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = '->> Setting {0} to {1}'
    # options defined in the cli take precedence over the ones in the ops file
    for k in ops0:
        v = ops0[k]
        n = dargs[k]
        if k in ['fast_disk', 'save_folder', 'save_path0']:
            if n:
                ops[k] = n
                print(set_param_msg.format(k, ops[k]))
        elif type(v) in [np.ndarray, list]:
            n = np.array(n)
            if np.any(n != np.array(v)):
                ops[k] = n.astype(type(v))
                print(set_param_msg.format(k, ops[k]))
        elif not v == type(v)(n):
            ops[k] = type(v)(n)
            print(set_param_msg.format(k, ops[k]))
    return args, ops


if __name__ == '__main__':
    args, ops = parse_args(add_args(argparse.ArgumentParser(description='Suite2p parameters')))
    if len(args.db) > 0:
        db = np.load(args.db, allow_pickle=True).item()
        from suite2p import run_s2p
        run_s2p(ops, db)
    else:
        from suite2p import gui
        gui.run()
