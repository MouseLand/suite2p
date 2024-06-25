"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import argparse
import numpy as np
from suite2p import default_ops, default_db, version


def add_args(parser: argparse.ArgumentParser):
    """
    Adds suite2p ops arguments to parser.
    """
    parser.add_argument("--single_plane", action="store_true",
                        help="run single plane db/ops")
    parser.add_argument("--ops", default=[], type=str, help="options")
    parser.add_argument("--db", default=[], type=str, help="options")
    parser.add_argument("--version", action="store_true", help="print version number.")
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = default_ops()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    ops = {**ops0, **ops}
    db = np.load(args.db, allow_pickle=True).item() if args.db else {}
    db = {**default_db(), **db}
    return args, db, ops

def main():
    args, db, ops = parse_args(
        add_args(argparse.ArgumentParser(description="Suite2p ops/db paths")))
    if args.version:
        print("suite2p v{}".format(version))
    elif args.single_plane and args.ops and args.db:
        from suite2p.run_s2p import run_plane, run_s2p
        # run single plane (does registration)
        run_plane(db=db, ops=ops, db_path=args.db)
    elif args.ops and args.db:
        run_s2p(db=db, ops=ops)
    else:
        from suite2p import gui
        gui.run()


if __name__ == "__main__":
    main()
