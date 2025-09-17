"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import argparse, os, platform
import numpy as np
from suite2p import default_settings, default_db, version
from suite2p.run_s2p import logger_setup, run_plane, run_s2p, get_save_folder
import logging

def add_args(parser: argparse.ArgumentParser):
    """
    Adds suite2p settings arguments to parser.
    """
    parser.add_argument("--single_plane", action="store_true",
                        help="run single plane db/settings")
    parser.add_argument("--settings", default=[], type=str, help="options")
    parser.add_argument("--db", default=[], type=str, help="options")
    parser.add_argument("--version", action="store_true", help="print version number.")
    parser.add_argument("--verbose", action="store_true", help="print more info during processing.")
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
    elif args.settings and args.db:
        if args.verbose:
            save_folder = db['save_path'] if args.single_plane else get_save_folder(db)
            logger_setup(save_folder)
        try:
            if args.single_plane:
                run_plane(db=db, settings=settings, db_path=args.db)
            else:
                run_s2p(db=db, settings=settings)
        except Exception as e:
            logging.exception(f'fatal error in {"run_plane" if args.single_plane else "run_s2p"}:')
            raise

    else:
        # Check if the OS is macOS and the machine is Apple Silicon (ARM-based)
        if platform.system() == "Darwin" and 'arm' in platform.processor().lower():
            # Set the number of threads for OpenMP and OpenBLAS
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            print("Environment set to use 1 thread for OpenMP and OpenBLAS (Apple Silicon macOS).")
        else:
            print("Not macOS on Apple Silicon, proceeding without limiting threads.")
            
        from suite2p import gui
        gui.run()#statfile="C:/DATA/exs2p/suite2p/plane0/stat.npy")


if __name__ == "__main__":
    main()
