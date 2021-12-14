from suite2p import run_s2p, default_ops
from pathlib import Path
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader

def main():
    
    #ops = default_ops() # populates ops with the default options
    #ops
    ops = default_ops()
    ops['input_format'] = "bruker_raw"
    ops['nchannels'] = 1
    ops['do_registration'] = 0
    ops['roidetect'] = 0
    ops['functional_chan'] = 2
    ops['fs'] = 15
    ops['block_size'] = [256, 256]
    ops['align_by_chan'] = 1
    ops['keep_movie_raw'] = 0
    
    print(ops)
    db = {
      'h5py': [], # a single h5 file path
      'h5py_key': 'data',
      'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
      'data_path' : ["Z:\\tbabola\\Experiments\\211207\\1x_quickmap_400um_5x_repeats_1_5xzoom_multisampling-036"],
      #'data_path': ['C:/Users/Travis/Dropbox (Kanoldlab)/PC/Desktop/Jade_GCaMP8_virus_no_multi-006_raw'], # a list of folders with tiffs 
                                             # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)                            
      'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
    }
    db['input_format'] = "bruker_raw"

    # run one experiment
    opsEnd = run_s2p(ops=ops, db=db)

def convertSingleSampling():
  #ops = default_ops() # populates ops with the default options
  #ops
  ops = default_ops()
  ops['input_format'] = "bruker_raw"
  ops['nchannels'] = 1
  ops['do_registration'] = 0
  ops['roidetect'] = 0
  ops['functional_chan'] = 2
  ops['fs'] = 15
  ops['block_size'] = [256, 256]
  ops['align_by_chan'] = 1
  ops['keep_movie_raw'] = 0

  print(ops)
  db = {
    'h5py': [], # a single h5 file path
    'h5py_key': 'data',
    'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
    'data_path' : ["Z:\\tbabola\\Experiments\\211203_QuickMap\\QuickMap-007"],
    #'data_path': ['C:/Users/Travis/Dropbox (Kanoldlab)/PC/Desktop/Jade_GCaMP8_virus_no_multi-006_raw'], # a list of folders with tiffs 
                                            # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)                            
    'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
  }
  db['input_format'] = "bruker_raw"

  # run one experiment
  opsEnd = run_s2p(ops=ops, db=db)


def loadBinaryToStack(dir):
  ops_file = list(raw_dir.glob("ops.npy"))[0]
  bin_file = list(raw_dir.glob("*.bin"))[0]
  ops = np.load(str(ops_file),allow_pickle=True).item()
  stack = np.fromfile(bin_file,'uint16')

  return stack.reshape(-1,ops["Lx"],ops["Ly"])

def loadTifToStack(dir):
  tifs = list(tif_path.glob("*.tif"))
  for i, tif in enumerate(tifs):
    with ScanImageTiffReader(str(tif)) as tiff:
      im = tiff.data()
    if i == 0:
      stack = np.zeros(shape=(len(tifs),im.shape[0],im.shape[1]),dtype=np.uint16)
    
    stack[i,:,:] = im

  return stack

if __name__ == "__main__":
    #main()
    # raw_dir = Path("Z:\\tbabola\\Experiments\\211207\\1x_quickmap_400um_5x_repeats_1_5xzoom_multisampling-036\\suite2p\\plane0")
    # raw_stack = loadBinaryToStack(raw_dir)

    # tif_path = Path("Z:\\tbabola\\Experiments\\211207\\1x_quickmap_400um_5x_repeats_1_5xzoom_multisampling-036 - Copy")
    # tif_stack = loadTifToStack(tif_path)
    # print("Raw and tiffs are equal for multisampling: {}".format(np.array_equal(tif_stack,raw_stack)))

    convertSingleSampling()
    raw_dir = Path("Z:\\tbabola\\Experiments\\211203_QuickMap\\QuickMap-007\\suite2p_orig\\plane0")
    raw_stack = loadBinaryToStack(raw_dir)

    raw_dir2 = Path("Z:\\tbabola\\Experiments\\211203_QuickMap\\QuickMap-007\\suite2p\\plane0")
    raw_stack2 = loadBinaryToStack(raw_dir2)

    print("Raw and raw stack are equal for single sampling: {}".format(np.array_equal(raw_stack, raw_stack2)))




    