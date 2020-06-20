import sys, getopt
from pathlib import Path


def invalid_argument():
    print("ERROR: Please provide a valid input argument.")
    print('registration_metrics.py -i <input_tif_file_path>')
    sys.exit(2)


def main(argv):
    input_path = ''
    input_file = None
    try:
        opts, args = getopt.getopt(argv, "hi:")
    except getopt.GetoptError:
        invalid_argument()
    # Go through command line arguments
    for opt, arg in opts:
        if opt == '-h':
            print('usage: registration_metrics.py -i <input_tif_file_path>')
            sys.exit()
        elif opt in '-i':
            input_path = arg
            input_file = Path(input_path)
    # Check if supplied input_file is
    if input_file and input_file.exists():
        print("Input file is : {}".format(input_file))
        print("Importing suite2p...")
        # Only import if input properly specified
        import suite2p
        default_ops = suite2p.default_ops()
        default_ops['do_regmetrics'] = True
        default_ops['roidetect'] = False
        default_ops['data_path'] = [input_file.parent]
        default_ops['tiff_list'] = [str(input_file.name)]
        default_ops['save_path0'] = '/Users/chriski/Desktop/suite2p_ws'
        print("Calculating registration metrics... ")
        suite2p.run_s2p(default_ops)
    else:
        invalid_argument()


if __name__ == "__main__":
   main(sys.argv[1:])