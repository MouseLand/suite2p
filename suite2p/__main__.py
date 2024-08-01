"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import argparse
import numpy as np
from suite2p import default_settings, default_db, version
from suite2p.run_s2p import run_plane, run_s2p

def add_args(parser: argparse.ArgumentParser):
    """
    Adds suite2p settings arguments to parser.
    """
    parser.add_argument("--single_plane", action="store_true",
                        help="run single plane db/settings")
    parser.add_argument("--settings", default=[], type=str, help="options")
    parser.add_argument("--db", default=[], type=str, help="options")
    parser.add_argument("--version", action="store_true", help="print version number.")
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns settings with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    settings0 = default_settings()
    settings = np.load(args.settings, allow_pickle=True).item() if args.settings else {}
    settings = {**settings0, **settings}
    db = np.load(args.db, allow_pickle=True).item() if args.db else {}
    db = {**default_db(), **db}
    return args, db, settings

def main():
    args, db, settings = parse_args(
        add_args(argparse.ArgumentParser(description="Suite2p settings/db paths")))
    if args.version:
        print("suite2p v{}".format(version))
    elif args.single_plane and args.settings and args.db:
        # run single plane (does registration)
        run_plane(db=db, settings=settings, db_path=args.db)
    elif args.settings and args.db:
        run_s2p(db=db, settings=settings)
    else:
        from suite2p import gui
        gui.run()#statfile="C:/DATA/exs2p/suite2p/plane0/stat.npy")


if __name__ == "__main__":
    main()
