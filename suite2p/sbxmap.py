import numpy as np
import os
import scipy.io as spio

def lmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
    
def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict
    
def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
     
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


class sbxmap(object):
    def __init__(self, filename):
        self.filename = os.path.splitext(filename)[0]

    @property
    def num_planes(self):
        if self.info['objective'] == 'meso_4mm' or self.info['objective'] == 'meso_10mm':
            if self.info['mesoscope']['roi_table'].ndim == 2:
                return self.info['mesoscope']['roi_table'].shape[0]
            else:
                return 1
        else:
            if 'otparam' in self.info:
                return self.info['otparam'][2] if self.info['otparam'] != [] else 1
            else:
                return 1
    @property
    def shape(self):
        if self.num_planes > 1:
            plane_length = self.info['length'] // self.num_planes
            #plane_length = len(np.arange(self.info['length'])[::self.num_planes])
            return (plane_length, self.info['sz'][0], self.info['sz'][1])
        else:
            return (self.info['length'], self.info['sz'][0], self.info['sz'][1])
    @property
    def info(self):
        _info = lmat(self.filename + '.mat')['info']
        # Fixes issue when using uint16 for memmapping
        _info['sz'] = _info['sz'].tolist()
        # Defining number of channels/size factor
        if _info['channels'] == -1:
            if _info['chan']['nchan'] == 2:
                _info['nChan'] = 2; factor = 1
            else:
                _info['nChan'] = 1; factor = 2
        if _info['channels'] == 1:
            _info['nChan'] = 2; factor = 1
        elif _info['channels'] == 2:
            _info['nChan'] = 1; factor = 2
        elif _info['channels'] == 3:
            _info['nChan'] = 1; factor = 2
        if _info['scanmode'] == 0:
            _info['recordsPerBuffer'] = _info['recordsPerBuffer']*2
        # Determine number of frames in whole file
        _info['length'] = int(
                os.path.getsize(self.filename + '.sbx')
                / _info['recordsPerBuffer']
                / _info['sz'][1]
                * factor
                / 4
                )
        _info['nSamples'] = _info['sz'][1] * _info['recordsPerBuffer'] * 2 * _info['nChan']
        return _info
    @property
    def channels(self):
        if self.info['channels'] == -1:
            sample = self.info['chan']['sample'][:2]
            if sample[0] and sample[1]:
                return ['green', 'red']
            if sample[0] and not sample[1]:
                return ['green']
            if not sample[0] and sample[1]:
                return['red']
        if self.info['channels'] == 1:
            return ['green', 'red']
        elif self.info['channels'] == 2:
            return ['green']
        elif self.info['channels'] == 3:
            return ['red']

    def data(self, length=[None], rows=[None], cols=[None]):
        fullshape = [self.info['nChan']*self.info['length']] + self.info['sz']
        mapped_data = np.memmap(self.filename + '.sbx', dtype='uint16')
        mapped_data = mapped_data.reshape(fullshape)
        return mapped_data

    def crop(self, length=[None], rows=[None], cols=[None], basename=None):
        basename = self.filename if basename is None else os.path.splitext(basename)[0]
        cropped_data = self.data(length=length, rows=rows, cols=cols)
        size = []
        for channel, planes in cropped_data.items():
            for plane, data in planes.items():
                size.append(np.prod(data.shape))
        size = np.sum(size)
        output_memmap = np.memmap('{}_cropped.sbx'.format(basename),
                                  dtype='uint16',
                                  shape=size,
                                  mode='w+'
                                  )
        spio_info = lmat.loadmat(self.filename + '.mat')
        spio_info['info']['sz'] = data.shape[1:]
        if rows is not [None]: # rows were cropped, update recordsPerBuffer
            spio_info['info']['originalRecordsPerBuffer'] = spio_info['info']['recordsPerBuffer'];
            spio_info['info']['recordsPerBuffer'] = spio_info['info']['sz'][0]
        lmat.spio.savemat('{}_cropped.mat'.format(basename), {'info':spio_info['info']})
        input_data = sbxmap('{}_cropped.sbx'.format(basename))
        for channel,channel_data in input_data.data().items():
            for plane,plane_data in channel_data.items():
                plane_data[:] = cropped_data[channel][plane]