"""
command-line utility tool to process tiffs to make them scanimage-compatible.

Usage Example:
    tiff2scanimage input.tif input-si.tif
"""

from tifffile import imread, imsave


def convert_tiff_to_scanimage(from_filename: str, to_filename: str) -> None:
    tif = imread(from_filename)
    imsave(to_filename, tif)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="convert tiff to ScanImage-compatible tiff.")
    parser.add_argument("infile", help="the tif file to process.")
    parser.add_argument("outfile", help="the tif filename to create")
    args = parser.parse_args()

    convert_tiff_to_scanimage(from_filename=args.infile, to_filename=args.outfile)


if __name__ == '__main__':
    main()