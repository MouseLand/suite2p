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
    from .run_s2p import default_ops
    ops0 = default_ops()
    for k in ops0.keys():
        v = dict(default = ops0[k],
                 help = '{0}: {1}'.format(k,ops0[k]))
        if k in ['fast_disk','save_folder','save_path0']:
            v['default'] = None
            v['type'] = str
        if type(v['default']) in [np.ndarray,list]:
            if len(v['default']):
                v['nargs'] = '+'
                v['type'] = type(v['default'][0])
        parser.add_argument('--'+k,**v)
    args = parser.parse_args()
    dargs = vars(args)
    ops = {}
    db= {}
    if len(args.ops)>0:
        ops = np.load(args.ops, allow_pickle=True).item()
    # options defined in the cli take precedence over the ones in the ops file
    for k in ops0:
        v = ops0[k]
        n = dargs[k]
        if k in ['fast_disk','save_folder','save_path0']:
            if not n is None:
                ops[k] = n
                print('->> Setting {0} to {1}'.format(k,ops[k]))
        elif type(v) in [np.ndarray,list]:            
            if len(n):
                n = np.array(n)
                if np.any(n != np.array(v)):
                    ops[k] = n.astype(type(v))
                    print('->> Setting {0} to {1}'.format(k,ops[k]))
        else:
            if not v == type(v)(n):
                ops[k] = type(v)(n)
                print('->> Setting {0} to {1}'.format(k,ops[k]))

    if len(args.db)>0:
        db = np.load(args.db, allow_pickle=True).item()
        from . import run_s2p
        run_s2p.run_s2p(ops, db)
    else:
        from .gui import gui2p
        gui2p.run()

    
if __name__ == '__main__':
    parse_arguments()
