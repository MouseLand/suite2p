from suite2p import run_s2p, default_ops

def main():
    
    #ops = default_ops() # populates ops with the default options
    #ops
    ops = default_ops()
    ops['input_format'] = "bruker_raw"
    ops['nchannels'] = 1
    ops['do_registration'] = 1
    ops['roidetect'] = 1
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
      'data_path' : ['Z:/temporary only/JL_DataAnalysis/20210924_m210319fR1cbaThy1_MF_HaltFreq_A1_1x15fps_270um-001'],
      #'data_path': ['C:/Users/Travis/Dropbox (Kanoldlab)/PC/Desktop/Jade_GCaMP8_virus_no_multi-006_raw'], # a list of folders with tiffs 
                                             # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)                            
      'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
    }
    db['input_format'] = "bruker_raw"

    # run one experiment
    opsEnd = run_s2p(ops=ops, db=db)



if __name__ == "__main__":
    main()
    