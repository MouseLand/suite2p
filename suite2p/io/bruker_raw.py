import gc
import glob
import json
from logging import raiseExceptions
import math
import os
import time
import sys
import xml.etree.ElementTree as ET

from typing import Union, Tuple, Optional
import numpy as np

from ScanImageTiffReader import ScanImageTiffReader
from tifffile import imread, TiffFile, TiffWriter, imsave
from pathlib import Path
from . import utils

def brukerRaw_to_binary(ops):
    """
    converts Bruker *.bin file for non-interleaved red channel recordings
    assumes SINGLE-PAGE tiffs where first channel has string 'Ch1'
    and also SINGLE FOLDER

    Parameters
    ----------
    ops : dictionary
        keys nplanes, nchannels, data_path, look_one_level_down, reg_file

    Returns
    -------
    ops : dictionary of first plane
        creates binaries ops['reg_file']
        assigns keys: tiffreader, first_tiffs, frames_per_folder, nframes, meanImg, meanImg_chan2
    """

    XMLfile = glob.glob(ops['data_path'][0]+"/*.xml")[0]
    if len(XMLfile) == 0:
        sys.exit("XML file not found. Process aborted.")

    ##parse XML file here, get parameters about pixel lines, etc, multisampling, numChannels, etc
    bruker_xml = parse_bruker_xml(XMLfile)
    nXpixels = int(bruker_xml['pixelsPerLine'])
    nYpixels = int(bruker_xml['linesPerFrame'])
    nframes_total = bruker_xml['nframes']
    ops['nframes'] = np.sum(nframes_total)
    nchannels = bruker_xml['nchannels']
    ncycles = bruker_xml['ncycles']
    ops['ncycles'] = ncycles
    functional_chan = ops['functional_chan']
    samplesPerPixel= int(bruker_xml['samplesPerPixel'])
    nplanes = bruker_xml['nplanes']
    ops['meanImg'] = np.zeros((nXpixels,nYpixels), np.float32)
    ops['meanImg_chan2'] = np.zeros((nXpixels,nYpixels), np.float32)

    if functional_chan > nchannels: ##just in case someone puts in 2 for functional channel
        functional_chan = 1

    if nchannels != ops['nchannels']:
         print("Number of channels input into Suite2p does not match number of channels in the XML file. Proceeding with XML parameters.")
         ops['nchannels'] = nchannels

    if nplanes != ops['nplanes']:
        print("Number of planes input into Suite2p does not match number of channels in the XML file. Proceeding with XML parameters.")
        ops['nplanes'] = nplanes

    #start timer
    t0 = time.time()

    # copy ops to list where each element is ops for each plane
    ops1 = utils.init_ops(ops)

    # open all binary files for writing
    ops1, fs, reg_file, reg_file_chan2 = find_bruker_raw_files_open_binaries(ops1)
    ops = ops1[0]

    #default behavior for cycles is just to jam them altogether, cycles also define the z-series time steps
    nframes_processed = 0
    total_frames_processed = 0
    leftover_samples = np.empty(shape=(0,),dtype=np.uint16)
    
    for i, f in enumerate(fs):
        ##for each raw file
        bin = np.fromfile(f,'uint16') - 2**13 #weird quirk when capturing with galvo-res scanner
        bin = np.concatenate((leftover_samples,bin))

        (completeFrames, bin, leftover_samples) = calculateCompleteFrames(bin, nXpixels,nYpixels,nchannels,samplesPerPixel, nplanes)
        if (completeFrames * nplanes + total_frames_processed) >  ops['nframes']:
            numToTake = ops['nframes'] - total_frames_processed
            bin = bin[:nXpixels*nYpixels*nchannels*samplesPerPixel*numToTake*nplanes]

        if samplesPerPixel > 1:
            bin = multisamplingAverage(bin,samplesPerPixel)
        elif samplesPerPixel == 1:
            bin[bin > 2**13] = 0

        for chan in range(nchannels):
            #grab appropriate samples and flip every other line
            bin_temp = bin[chan::nchannels]
            bin_temp = bin_temp.reshape(-1,nXpixels,nYpixels)
            bin_temp[:,1::2,:] = np.flip(bin_temp[:,1::2,:],2)

            if ops['keep_movie_raw'] == 1:
                    savestr = "/Ch{}_{}.tiff".format(chan,i)
                    imsave(ops['data_path'][0]+savestr,bin_temp)

            for plane in np.arange(0, nplanes):
                bin_temp_plane = bin_temp[0::nplanes,:,:]
                meanImg = np.sum(bin_temp_plane,0)
                if (chan+1) == functional_chan: #s2p thinks it's yummy when first channel is used for functional
                    reg_file[plane].write(bytearray(bin_temp_plane))
                    total_frames_processed = total_frames_processed + bin_temp_plane.shape[0]
                    ops['meanImg'] += meanImg
                else: #could probably be upgraded to include more channels, but fine for now
                    reg_file_chan2[plane].write(bytearray(bin_temp))
                    ops['meanImg_chan2'] += meanImg

        print('%d frames of binary, time %0.2f sec.'%(total_frames_processed,time.time()-t0))


    # write ops files
    do_registration = ops['do_registration']
    for ops in ops1:
        ops['Ly'], ops['Lx'] = nYpixels, nXpixels
        if not do_registration:
            ops['yrange'] = np.array([0,ops['Ly']])
            ops['xrange'] = np.array([0,ops['Lx']])
        ops['meanImg'] /= ops['nframes']
        if nchannels>1:
            ops['meanImg_chan2'] /= ops['nframes']
        np.save(ops['ops_path'], ops)
    # close all binary files and write ops files
    for j in range(0,nplanes):
        reg_file[j].close()
        if nchannels>1:
            reg_file_chan2[j].close()
    return ops1[0]

def multisamplingAverage(bin, samplesPerPixel):
    bin = bin.reshape(-1,samplesPerPixel)
    addmask = np.sum(bin <= 2**13,1,dtype="uint16")
    bin[bin > 2**13] = 0
    bin = np.floor_divide(np.sum(bin,1,dtype="uint16"),addmask)

    return bin

def calculateCompleteFrames(bin, nXpixels,nYpixels,nchannels,samplesPerPixel, nplanes):
    #calculate the number of complete frames from the samples read in
    complete_frames = np.floor(bin.shape[0]/(nXpixels*nYpixels*nchannels*samplesPerPixel*nplanes)).astype(int)

    #get samples for complete frames
    samples = bin[:nXpixels*nYpixels*nchannels*samplesPerPixel*complete_frames*nplanes]

    #samples leftover in buffer for next frame
    leftover_samples = bin[nXpixels*nYpixels*nchannels*samplesPerPixel*complete_frames*nplanes:]

    return (complete_frames, samples, leftover_samples)

def parse_bruker_xml(xmlfile):
    #wondering weather or not to pull just the relevant information out
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    b_xml = {}
    for thing in root.findall("./PVStateShard/"):
        try:
            b_xml[thing.attrib['key']] = thing.attrib['value']
        except:
            pass

    for thing in root.findall("Sequence"):
        try:
            print(thing.attrib['key']) 
            print(thing.attrib['value'])
        except:
            pass
    
    #if this is a time series with z-stack
    if root.findall("Sequence")[0].attrib['type'] == 'TSeries ZSeries Element':
        b_xml['bidirectional'] = root.findall("Sequence")[0].attrib['bidirectionalZ']
        b_xml['nplanes'] = len(root.findall("Sequence")[0]) - 1
    elif root.findall("Sequence")[0].attrib['type'] == 'TSeries Timed Element':
        b_xml['nplanes'] = 1
    
    b_xml['ncycles'] = len(root) - 2
    b_xml['nframes'] = [len(root[i+2].findall('Frame')) for i in range(b_xml['ncycles'])] #not very pythonic, but just counts frames per cycle
    b_xml['nchannels'] = len(root[2].findall('Frame')[0].findall('File')) 
    return b_xml

def find_bruker_raw_files_open_binaries(ops1):
    """  finds bruker raw files and opens binaries for writing
        Parameters
        ----------
        ops1 : list of dictionaries
        'keep_movie_raw', 'data_path', 'look_one_level_down', 'reg_file'...

        Returns
        -------
            ops1 : list of dictionaries
                adds fields 'filelist', 'first_tiffs', opens binaries

    """
    reg_file = []
    reg_file_chan2=[]

    for ops in ops1:
        nchannels = ops['nchannels']
        if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
            reg_file.append(open(ops['raw_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['raw_file_chan2'], 'wb'))
        else:
            reg_file.append(open(ops['reg_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))

    fs = list(Path(ops['data_path'][0]).glob("*RAWDATA*"))

    for ops in ops1:
        ops['filelist'] = fs

    return ops1, fs, reg_file, reg_file_chan2