"""
Copyright © 2023 Yoav Livneh Lab, Authored by Yael Prilutski.
"""

import numpy as np

from os import makedirs, listdir
from os.path import isdir, isfile, getsize, join

try:
    from xmltodict import parse
    HAS_XML = True
except (ModuleNotFoundError, ImportError):
    HAS_XML = False

EXTENSION = 'raw'


def raw_to_binary(ops, use_recorded_defaults=True):

    """ Finds RAW files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        "data_path"

    use_recorded_defaults : bool
        Recorded session parameters are used when 'True',
        otherwise |ops| is expected to contain the following (additional) keys:
          "nplanes",
          "nchannels",
          "fs"

    Returns
    -------
        ops : dictionary of first plane

    """

    if not HAS_XML:
        raise ImportError("xmltodict is required for RAW file support (pip install xmltodict)")

    # Load raw file configurations
    raw_file_configurations = [_RawFile(path) for path in ops['data_path']]

    # Split ops by captured planes
    ops_paths = _initialize_destination_files(ops, raw_file_configurations, use_recorded_defaults=use_recorded_defaults)

    # Convert all runs in order
    for path in ops['data_path']:
        print(f'Converting raw to binary: `{path}`')
        ops_loaded = [np.load(i, allow_pickle=True)[()] for i in ops_paths]
        _raw2bin(ops_loaded, _RawFile(path))

    # Reload edited ops
    ops_loaded = [np.load(i, allow_pickle=True)[()] for i in ops_paths]

    # Create a mean image with the final number of frames
    _update_mean(ops_loaded)

    # Load & return all ops
    return ops_loaded[0]


def _initialize_destination_files(ops, raw_file_configurations, use_recorded_defaults=True):

    """ Prepares raw2bin conversion environment (files & folders) """

    configurations = [
        [cfg.channel, cfg.zplanes, cfg.xpx, cfg.ypx, cfg.frame_rate, cfg.xsize, cfg.ysize]
        for cfg in raw_file_configurations
    ]

    # Make sure all ops match each other
    assert all(conf == configurations[0] for conf in configurations), \
        f'Data attributes do not match. Can not concatenate shapes: {[conf for conf in configurations]}'

    # Load configuration from first file in paths
    cfg = raw_file_configurations[0]

    # Expand configuration from defaults when necessary
    if use_recorded_defaults:
        ops['nplanes'] = cfg.zplanes
        if cfg.channel > 1:
            ops['nchannels'] = 2
        ops['fs'] = cfg.frame_rate

    # Prepare conversion environment for all files
    ops_paths = []
    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    second_plane = False
    for i in range(0, nplanes):
        ops['save_path'] = join(ops['save_path0'], 'suite2p', f'plane{i}')

        if ('fast_disk' not in ops) or len(ops['fast_disk']) == 0 or second_plane:
            ops['fast_disk'] = ops['save_path']
            second_plane = True
        else:
            ops['fast_disk'] = join(ops['fast_disk'], 'suite2p', f'plane{i}')

        ops['ops_path'] = join(ops['save_path'], 'ops.npy')
        ops['reg_file'] = join(ops['fast_disk'], 'data.bin')
        isdir(ops['fast_disk']) or makedirs(ops['fast_disk'])
        isdir(ops['save_path']) or makedirs(ops['save_path'])
        open(ops['reg_file'], 'wb').close()
        if nchannels > 1:
            ops['reg_file_chan2'] = join(ops['fast_disk'], 'data_chan2.bin')
            open(ops['reg_file_chan2'], 'wb').close()

        ops['meanImg'] = np.zeros((cfg.xpx, cfg.ypx), np.float32)
        ops['nframes'] = 0
        ops['frames_per_run'] = []
        if nchannels > 1:
            ops['meanImg_chan2'] = np.zeros((cfg.xpx, cfg.ypx), np.float32)

        # write ops files
        do_registration = ops['do_registration']
        ops['Ly'] = cfg.xpx
        ops['Lx'] = cfg.ypx
        if not do_registration:
            ops['yrange'] = np.array([0, ops['Ly']])
            ops['xrange'] = np.array([0, ops['Lx']])

        ops_paths.append(ops['ops_path'])
        np.save(ops['ops_path'], ops)

    # Environment ready;
    return ops_paths


def _raw2bin(all_ops, cfg):

    """ Converts a single RAW file to BIN format """

    frames_in_chunk = int(all_ops[0]['batch_size'])

    with open(cfg.path, 'rb') as raw_file:
        chunk = frames_in_chunk * cfg.xpx * cfg.ypx * cfg.channel * cfg.recorded_planes * 2
        raw_data_chunk = raw_file.read(chunk)
        while raw_data_chunk:
            data = np.frombuffer(raw_data_chunk, dtype=np.int16)
            current_frames = int(len(data) / cfg.xpx / cfg.ypx / cfg.recorded_planes)

            if cfg.channel > 1:
                channel_a, channel_b = _split_into_2_channels(data.reshape(
                    current_frames * cfg.recorded_planes, cfg.xpx, cfg.ypx))
                reshaped_data = []
                for i in range(cfg.recorded_planes):
                    channel_a_plane = channel_a[i::cfg.recorded_planes]
                    channel_b_plane = channel_b[i::cfg.recorded_planes]
                    reshaped_data.append([channel_a_plane, channel_b_plane])

            else:
                reshaped_data = data.reshape(cfg.recorded_planes, current_frames, cfg.xpx, cfg.ypx)

            for plane in range(0, cfg.zplanes):
                ops = all_ops[plane]
                plane_data = reshaped_data[plane]

                if cfg.channel > 1:
                    with open(ops['reg_file'], 'ab') as bin_file:
                        bin_file.write(bytearray(plane_data[0].astype(np.int16)))
                    with open(ops['reg_file_chan2'], 'ab') as bin_file2:
                        bin_file2.write(bytearray(plane_data[1].astype(np.int16)))
                    ops['meanImg'] += plane_data[0].astype(np.float32).sum(axis=0)
                    ops['meanImg_chan2'] = ops['meanImg_chan2'] + plane_data[1].astype(np.float32).sum(axis=0)

                else:
                    with open(ops['reg_file'], 'ab') as bin_file:
                        bin_file.write(bytearray(plane_data.astype(np.int16)))
                    ops['meanImg'] = ops['meanImg'] + plane_data.astype(np.float32).sum(axis=0)

            raw_data_chunk = raw_file.read(chunk)

    for ops in all_ops:
        total_frames = int(cfg.size / cfg.xpx / cfg.ypx / cfg.recorded_planes / cfg.channel / 2)
        ops['frames_per_run'].append(total_frames)
        ops['nframes'] += total_frames
        np.save(ops['ops_path'], ops)


def _split_into_2_channels(data):

    """ Utility function, used during conversion - splits given raw data into 2 separate channels """

    frames = data.shape[0]
    channel_a_index = list(filter(lambda x: x % 2 == 0, range(frames)))
    channel_b_index = list(filter(lambda x: x % 2 != 0, range(frames)))
    return data[channel_a_index], data[channel_b_index]


def _update_mean(ops_loaded):

    """ Adjusts all "meanImg" values at the end of raw-to-binary conversion. """

    for ops in ops_loaded:
        ops['meanImg'] /= ops['nframes']
        np.save(ops['ops_path'], ops)


class _RawConfig:

    """ Handles XML configuration parsing and exposes video shape & parameters for Thorlabs RAW files """

    def __init__(self, raw_file_size, xml_path):

        assert isfile(xml_path)

        self._xml_path = xml_path

        self.zplanes = 1
        self.recorded_planes = 1

        self.xpx = None
        self.ypx = None
        self.channel = None
        self.frame_rate = None
        self.xsize = None
        self.ysize = None
        self.nframes = None

        # Load configuration defaults
        with open(self._xml_path, 'r', encoding='utf-8') as file:
            self._load_xml_config(raw_file_size, parse(file.read()))

        # Make sure all fields have been filled
        assert None not in (self.xpx, self.ypx, self.channel, self.frame_rate, self.xsize, self.ysize, self.nframes)

        # Extract data shape
        self._shape = self._find_shape()

    @property
    def shape(self): return self._shape

    def _find_shape(self):

        """ Discovers data dimensions """

        shape = [self.nframes, self.xpx, self.ypx]
        if self.recorded_planes > 1:
            shape.insert(0, self.recorded_planes)
        if self.channel > 1:
            shape[0] = self.nframes * 2
        return shape

    def _load_xml_config(self, raw_file_size, xml):

        """ Loads recording parameters from attached XML;

        :param raw_file_size: Size (in bytes) of main RAW file
        :param xml: Original XML contents as created during data acquisition (pre-parsed to a python dictionary) """

        xml_data = xml['ThorImageExperiment']

        self.xpx = int(xml_data['LSM']['@pixelX'])
        self.ypx = int(xml_data['LSM']['@pixelY'])
        self.channel = int(xml_data['LSM']['@channel'])
        self.frame_rate = float(xml_data['LSM']['@frameRate'])
        self.xsize = float(xml_data['LSM']['@widthUM'])
        self.ysize = float(xml_data['LSM']['@heightUM'])
        self.nframes = int(xml_data['Streaming']['@frames'])

        flyback = int(xml_data['Streaming']['@flybackFrames'])
        zenable = int(xml_data['Streaming']['@zFastEnable'])
        planes = int(xml_data['ZStage']['@steps'])

        if self.channel > 1:
            self.channel = 2

        if zenable > 0:
            self.zplanes = planes
            self.recorded_planes = flyback + self.zplanes
            self.nframes = int(self.nframes / self.recorded_planes)

        if xml_data['ExperimentStatus']['@value'] == 'Stopped':
            # Recording stopped in the middle, the written frame number isn't correct
            all_frames = int(raw_file_size / self.xpx / self.ypx / self.recorded_planes / self.channel / 2)
            self.nframes = int(all_frames / self.recorded_planes)


class _RawFile(_RawConfig):

    """ These objects represents all recording parameters per single Thorlabs RAW file """

    _MAIN_FILE_SUFFIX = f'001.{EXTENSION}'

    def __init__(self, dir_name):
        self._dirname = dir_name
        filenames = listdir(dir_name)

        # Find main raw file
        main_files = [fn for fn in filenames if fn.lower().endswith(self._MAIN_FILE_SUFFIX)]
        assert 1 == len(main_files), f'Corrupted directory structure: "{dir_name}"'
        self._raw_file_path = join(dir_name, main_files[0])
        self._raw_file_size = getsize(self._raw_file_path)

        # Load XML config
        xml_files = [fn for fn in filenames if fn.lower().endswith('.xml')]
        assert 1 == len(xml_files), f'Missing required XML configuration file from dir="{dir_name}"'
        _RawConfig.__init__(self, self._raw_file_size, join(dir_name, xml_files[0]))

    @property
    def path(self): return self._raw_file_path

    @property
    def size(self): return self._raw_file_size
