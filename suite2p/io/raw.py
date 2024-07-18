"""
Copyright © 2023 Yoav Livneh Lab, Authored by Yael Prilutski.
"""

import numpy as np

from os import makedirs, listdir
from os.path import isdir, isfile, getsize, join
import logging 
logger = logging.getLogger(__name__)

try:
    from xmltodict import parse
    HAS_XML = True
except (ModuleNotFoundError, ImportError):
    HAS_XML = False

EXTENSION = 'raw'


def raw_to_binary(settings, use_recorded_defaults=True):

    """ Finds RAW files and writes them to binaries

    Parameters
    ----------
    settings : dictionary
        "data_path"

    use_recorded_defaults : bool
        Recorded session parameters are used when 'True',
        otherwise |settings| is expected to contain the following (additional) keys:
          "nplanes",
          "nchannels",
          "fs"

    Returns
    -------
        settings : dictionary of first plane

    """

    if not HAS_XML:
        raise ImportError("xmltodict is required for RAW file support (pip install xmltodict)")

    # Load raw file configurations
    raw_file_configurations = [_RawFile(path) for path in settings['data_path']]

    # Split settings by captured planes
    settings_paths = _initialize_destination_files(settings, raw_file_configurations, use_recorded_defaults=use_recorded_defaults)

    # Convert all runs in order
    for path in settings['data_path']:
        logger.info(f'Converting raw to binary: `{path}`')
        settings_loaded = [np.load(i, allow_pickle=True)[()] for i in settings_paths]
        _raw2bin(settings_loaded, _RawFile(path))

    # Reload edited settings
    settings_loaded = [np.load(i, allow_pickle=True)[()] for i in settings_paths]

    # Create a mean image with the final number of frames
    _update_mean(settings_loaded)

    # Load & return all settings
    return settings_loaded[0]


def _initialize_destination_files(settings, raw_file_configurations, use_recorded_defaults=True):

    """ Prepares raw2bin conversion environment (files & folders) """

    configurations = [
        [cfg.channel, cfg.zplanes, cfg.xpx, cfg.ypx, cfg.frame_rate, cfg.xsize, cfg.ysize]
        for cfg in raw_file_configurations
    ]

    # Make sure all settings match each other
    assert all(conf == configurations[0] for conf in configurations), \
        f'Data attributes do not match. Can not concatenate shapes: {[conf for conf in configurations]}'

    # Load configuration from first file in paths
    cfg = raw_file_configurations[0]

    # Expand configuration from defaults when necessary
    if use_recorded_defaults:
        settings['nplanes'] = cfg.zplanes
        if cfg.channel > 1:
            settings['nchannels'] = 2
        settings['fs'] = cfg.frame_rate

    # Prepare conversion environment for all files
    settings_paths = []
    nplanes = settings['nplanes']
    nchannels = settings['nchannels']
    second_plane = False
    for i in range(0, nplanes):
        settings['save_path'] = join(settings['save_path0'], 'suite2p', f'plane{i}')

        if ('fast_disk' not in settings) or len(settings['fast_disk']) == 0 or second_plane:
            settings['fast_disk'] = settings['save_path']
            second_plane = True
        else:
            settings['fast_disk'] = join(settings['fast_disk'], 'suite2p', f'plane{i}')

        settings['settings_path'] = join(settings['save_path'], 'settings.npy')
        settings['reg_file'] = join(settings['fast_disk'], 'data.bin')
        isdir(settings['fast_disk']) or makedirs(settings['fast_disk'])
        isdir(settings['save_path']) or makedirs(settings['save_path'])
        open(settings['reg_file'], 'wb').close()
        if nchannels > 1:
            settings['reg_file_chan2'] = join(settings['fast_disk'], 'data_chan2.bin')
            open(settings['reg_file_chan2'], 'wb').close()

        settings['meanImg'] = np.zeros((cfg.xpx, cfg.ypx), np.float32)
        settings['nframes'] = 0
        settings['frames_per_run'] = []
        if nchannels > 1:
            settings['meanImg_chan2'] = np.zeros((cfg.xpx, cfg.ypx), np.float32)

        # write settings files
        do_registration = settings['do_registration']
        settings['Ly'] = cfg.xpx
        settings['Lx'] = cfg.ypx
        if not do_registration:
            settings['yrange'] = np.array([0, settings['Ly']])
            settings['xrange'] = np.array([0, settings['Lx']])

        settings_paths.append(settings['settings_path'])
        np.save(settings['settings_path'], settings)

    # Environment ready;
    return settings_paths


def _raw2bin(all_settings, cfg):

    """ Converts a single RAW file to BIN format """

    frames_in_chunk = int(all_settings[0]['batch_size'])

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
                settings = all_settings[plane]
                plane_data = reshaped_data[plane]

                if cfg.channel > 1:
                    with open(settings['reg_file'], 'ab') as bin_file:
                        bin_file.write(bytearray(plane_data[0].astype(np.int16)))
                    with open(settings['reg_file_chan2'], 'ab') as bin_file2:
                        bin_file2.write(bytearray(plane_data[1].astype(np.int16)))
                    settings['meanImg'] += plane_data[0].astype(np.float32).sum(axis=0)
                    settings['meanImg_chan2'] = settings['meanImg_chan2'] + plane_data[1].astype(np.float32).sum(axis=0)

                else:
                    with open(settings['reg_file'], 'ab') as bin_file:
                        bin_file.write(bytearray(plane_data.astype(np.int16)))
                    settings['meanImg'] = settings['meanImg'] + plane_data.astype(np.float32).sum(axis=0)

            raw_data_chunk = raw_file.read(chunk)

    for settings in all_settings:
        total_frames = int(cfg.size / cfg.xpx / cfg.ypx / cfg.recorded_planes / cfg.channel / 2)
        settings['frames_per_run'].append(total_frames)
        settings['nframes'] += total_frames
        np.save(settings['settings_path'], settings)


def _split_into_2_channels(data):

    """ Utility function, used during conversion - splits given raw data into 2 separate channels """

    frames = data.shape[0]
    channel_a_index = list(filter(lambda x: x % 2 == 0, range(frames)))
    channel_b_index = list(filter(lambda x: x % 2 != 0, range(frames)))
    return data[channel_a_index], data[channel_b_index]


def _update_mean(settings_loaded):

    """ Adjusts all "meanImg" values at the end of raw-to-binary conversion. """

    for settings in settings_loaded:
        settings['meanImg'] /= settings['nframes']
        np.save(settings['settings_path'], settings)


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
