import argparse
import suite2p
import numpy as np
from typing import NamedTuple
from pathlib import Path
from suite2p.__main__ import add_args, parse_args


class RegMetricResult(NamedTuple):
    nplane: int
    avg_offs: np.array
    max_offs: np.array


def registration_metrics(data_path, tiff_list, ops, nPC=10):
    """
    Displays registration offsets calculated on pclow and pchigh frames. If registration was performed well,
    the PCs should not contain movement. All offsets calculated on pclow/pchigh frames should be close to zero.
    """
    ops['do_regmetrics'] = True
    ops['roidetect'] = False
    ops['reg_metric_n_pc'] = nPC
    ops['data_path'] = data_path
    if tiff_list:
        ops['tiff_list'] = tiff_list

    result_ops = suite2p.run_s2p(ops)
    metric_results = []
    for nplane, result_op in enumerate(result_ops):
        offsets = result_op['regDX']
        result = RegMetricResult(nplane=nplane, avg_offs=np.mean(offsets, axis=0), max_offs=np.max(offsets, axis=0))
        metric_results.append(result)
    return metric_results


def main():

    default_parser = add_args(argparse.ArgumentParser(description='Suite2p parameters'))
    default_parser.add_argument('data_path', type=str, nargs=1, help='Path to directory with input files')
    default_parser.add_argument('--tiff_list', default=[], type=str, nargs='*', help='Input files selected')
    default_parser.add_argument('--n_pc', default=10, type=int, help='Number of PCs')
    args, ops = parse_args(default_parser)
    reg_metric_results = registration_metrics(
        data_path=args.data_path, tiff_list=args.tiff_list, ops=ops, nPC=args.n_pc
    )
    for r in reg_metric_results:
        print(
            f"""
            Plane {r.nplane}:
            Avg_Rigid: {r.avg_offs[0]:.6f} \tAvg_Average NR: {r.avg_offs[1]:.6f} \tAvg_Max NR: {r.avg_offs[2]:.6f}
            Max_Rigid: {r.max_offs[0]:.6f} \tMax_Average NR: {r.max_offs[1]:.6f} \tMax_Max NR: {r.max_offs[2]:.6f}
            """.replace('            ', '')
        )


if __name__ == "__main__":
    main()