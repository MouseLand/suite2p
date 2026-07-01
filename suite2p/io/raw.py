"""
Copyright © 2023 Yoav Livneh Lab, Authored by Yael Prilutski.
"""

import os
import numpy as np

from os import listdir
from os.path import isfile, getsize, join

try:
    from xmltodict import parse
    HAS_XML = True
except (ModuleNotFoundError, ImportError):
    HAS_XML = False

EXTENSION = 'raw'


def raw_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """Finds RAW files and writes them to binaries.

    Parameters
    ----------
    dbs : list of dict
        Per-plane database dictionaries. Must contain "file_list", "nplanes",
        "nchannels", "batch_size", "functional_chan". Updated in-place with
        "Ly", "Lx", "nframes", "frames_per_folder", "meanImg", and "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels > 1).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries.
    """
    if not HAS_XML:
        raise ImportError("xmltodict is required for RAW file support (pip install xmltodict)")

    rawlist = dbs[0]["file_list"]
    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    nfunc = dbs[0]["functional_chan"] - 1 if nchannels > 1 else 0
    frames_in_chunk = int(dbs[0]["batch_size"])

    for j in range(nplanes):
        dbs[j]["nframes"] = 0
        dbs[j]["frames_per_folder"] = np.zeros(len(rawlist), np.int32)

    iall = 0
    for ifile, raw_path in enumerate(rawlist):
        cfg = _RawFile(raw_path)
        chunk_bytes = frames_in_chunk * cfg.xpx * cfg.ypx * cfg.channel * cfg.recorded_planes * 2

        with open(cfg.path, 'rb') as f:
            raw_chunk = f.read(chunk_bytes)
            while raw_chunk:
                data = (np.frombuffer(raw_chunk, dtype=np.uint16) // 2).astype(np.int16)
                current_frames = int(len(data) / cfg.xpx / cfg.ypx / cfg.recorded_planes)

                if cfg.channel > 1:
                    channel_a, channel_b = _split_into_2_channels(
                        data.reshape(current_frames * cfg.recorded_planes, cfg.xpx, cfg.ypx))
                    reshaped = [[channel_a[i::cfg.recorded_planes],
                                 channel_b[i::cfg.recorded_planes]]
                                for i in range(cfg.recorded_planes)]
                else:
                    reshaped = data.reshape(current_frames, cfg.recorded_planes, cfg.xpx, cfg.ypx)

                for j in range(nplanes):
                    if iall == 0:
                        dbs[j]["Ly"] = cfg.xpx
                        dbs[j]["Lx"] = cfg.ypx
                        dbs[j]["meanImg"] = np.zeros((cfg.xpx, cfg.ypx), np.float32)
                        if nchannels > 1:
                            dbs[j]["meanImg_chan2"] = np.zeros((cfg.xpx, cfg.ypx), np.float32)

                    if cfg.channel > 1:
                        ch_func = reshaped[j][nfunc].astype(np.int16)
                        ch_other = reshaped[j][1 - nfunc].astype(np.int16)
                        reg_file[j].write(bytearray(ch_func))
                        reg_file_chan2[j].write(bytearray(ch_other))
                        dbs[j]["meanImg"] += ch_func.astype(np.float32).sum(axis=0)
                        dbs[j]["meanImg_chan2"] += ch_other.astype(np.float32).sum(axis=0)
                        nframes_chunk = ch_func.shape[0]
                    else:
                        plane_data = reshaped[:, j, :, :].astype(np.int16)
                        reg_file[j].write(bytearray(plane_data))
                        dbs[j]["meanImg"] += plane_data.astype(np.float32).sum(axis=0)
                        nframes_chunk = plane_data.shape[0]

                    dbs[j]["nframes"] += nframes_chunk
                    dbs[j]["frames_per_folder"][ifile] += nframes_chunk

                iall += current_frames
                raw_chunk = f.read(chunk_bytes)

    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs


def _split_into_2_channels(data):

    """ Utility function, used during conversion - splits given raw data into 2 separate channels """

    frames = data.shape[0]
    channel_a_index = list(filter(lambda x: x % 2 == 0, range(frames)))
    channel_b_index = list(filter(lambda x: x % 2 != 0, range(frames)))
    return data[channel_a_index], data[channel_b_index]


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

    def __init__(self, path):
        # Accept either a .raw file path or a session directory
        if os.path.isfile(path):
            self._raw_file_path = path
            self._dirname = os.path.dirname(path)
        else:
            self._dirname = path
            filenames = listdir(path)
            raw_files = [fn for fn in filenames if fn.lower().endswith(f'.{EXTENSION}')]
            assert 1 == len(raw_files), f'Expected one .raw file in "{path}", found {len(raw_files)}'
            self._raw_file_path = join(path, raw_files[0])

        self._raw_file_size = getsize(self._raw_file_path)

        # Load XML config from the same directory
        filenames = listdir(self._dirname)
        xml_files = [fn for fn in filenames if fn.lower().endswith('.xml')]
        assert 1 == len(xml_files), f'Missing required XML configuration file from dir="{self._dirname}"'
        _RawConfig.__init__(self, self._raw_file_size, join(self._dirname, xml_files[0]))

    @property
    def path(self): return self._raw_file_path

    @property
    def size(self): return self._raw_file_size
