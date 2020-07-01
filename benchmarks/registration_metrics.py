import click
from pathlib import Path


@click.command()
@click.argument('inputfile', type=click.Path(exists=True))
@click.argument('outputpath', type=click.Path(exists=True))
@click.option('-nc', '--nchannels', default=1, show_default=True, help="Number of channels")
@click.option('-np', '--nplanes', default=1, show_default=True, help="Number of planes")
@click.option('-op', '--one_preg', default=False, show_default=True,
              help="Whether to perform high-pass filtering and tapering")
@click.option('-bdc', '--bidi_corrected', default=False, show_default=True, help="Bidirectional phase offset correction")
@click.option('-ss', '--smooth_sigma', default=1.15, show_default=True, help="Sigma for smoothing in time")
@click.option('-shp', '--spatial_hp_reg', default=26, show_default=True, help="Window for spatial high-pass filtering")
@click.option('-ps', '--pre_smooth', default=1, show_default=True,
              help="Whether to smooth before high-pass filtering before registration")
@click.option('-st', '--spatial_taper', default=50, show_default=True,
              help="How much to ignore on edges (important for vignetted windows,for FFT padding do not set BELOW 3*ops['smooth_sigma']")
def run_pc_reg_metrics(inputfile, outputpath, nchannels, nplanes, one_preg,
                       bidi_corrected, smooth_sigma, spatial_hp_reg, pre_smooth, spatial_taper):
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
    # Assign necessary parameters to default ops for suite2p
    default_ops = suite2p.default_ops()
    default_ops['do_regmetrics'] = True
    default_ops['roidetect'] = False
    default_ops['data_path'] = [Path(inputfile).parent]
    default_ops['tiff_list'] = [inputfile]
    default_ops['save_path0'] = outputpath
    # Assign optional ops for suite2p
    default_ops['nplanes'] = nplanes
    default_ops['nchannels'] = nchannels
    default_ops['1Preg'] = one_preg
    default_ops['bidi_corrected'] = bidi_corrected
    default_ops['smooth_sigma'] = smooth_sigma
    default_ops['spatial_hp_reg'] = spatial_hp_reg
    default_ops['pre_smooth'] = pre_smooth
    default_ops['spatial_taper'] = spatial_taper
    click.echo("Calculating registration metrics...")
    result_ops = suite2p.run_s2p(default_ops)
    for plane in range(nplanes):
        dx = result_ops[plane]['regDX']
        reg_pc =result_ops[plane]['regPC']
        tpc = result_ops[plane]['tPC']
        bottom_pc = reg_pc[0]
        top_pc = reg_pc[1]
        click.echo(bottom_pc.shape)
        click.echo(top_pc.shape)


if __name__ == "__main__":
    run_pc_reg_metrics()