import click
from pathlib import Path


@click.command()
@click.argument('inputfile', type=click.Path(exists=True))
@click.argument('outputpath', type=click.Path(exists=True))
def run_pc_reg_metrics(inputfile, outputpath):
    """
    Returns registration PC statistics for given INPUTFILE. Stores output of suite2p pipeline in
    OUTPUTPATH.

    \b
    Arguments:
        INPUTFILE is the tif you want to calculate registration metrics for.
        OUTPUTHPATH is the path where the output of suite2p is stored.
    """
    click.echo("Input file is: {} ".format(click.format_filename(inputfile)))
    click.echo("Output path is: {}".format(click.format_filename(outputpath)))
    click.echo("Importing suite2p...")
    import suite2p
    # Assign default ops for suite2p
    default_ops = suite2p.default_ops()
    default_ops['do_regmetrics'] = True
    default_ops['roidetect'] = False
    default_ops['data_path'] = [Path(inputfile).parent]
    default_ops['tiff_list'] = [inputfile]
    default_ops['save_path0'] = outputpath
    click.echo("Calculating registration metrics...")
    result_ops = suite2p.run_s2p(default_ops)


if __name__ == "__main__":
    run_pc_reg_metrics()