import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def run_multiday_singleday(db_exp, root='/media/carsen/disk2/CA1_tracking/'):
    from suite2p import run_s2p, default_settings, default_db, parameters
    from suite2p.run_s2p import logger_setup
    from suite2p.io import BinaryFile
    from suite2p.detection import bin_movie

    root = Path(root)
    
    db = default_db()
    settings = default_settings()

    data_path = [str(root / f"{exp['mname']}/{exp['datexp']}/{exp['blk']}") for exp in db_exp]
    import json
    with open(Path(data_path[0]) / "ops.json", "r") as f:
        settings_in = json.load(f)
    print(settings_in)

    db, settings, settings_in = parameters.convert_settings_orig(settings_in, db=db, settings=settings)

    db['data_path'] = data_path
    db['save_path0'] = str(Path(root / db_exp[0]['mname']))
    db['keep_movie_raw'] = False
    settings['classification']['preclassify'] = 0.0
    settings['run']['multiplane_parallel'] = 0
    settings['detection']['sparsery_settings']['max_ROIs'] = 20000

    run_s2p(db=db, settings=settings)

    data_path_all = [str(root / f"{exp['mname']}/{exp['datexp']}/{exp['blk']}") for exp in db_exp]

    for j in range(len(data_path_all)):
        db = default_db()
        settings = default_settings()

        import json
        with open(Path(data_path_all[j]) / "ops.json", "r") as f:
            settings_in = json.load(f)
        print(settings_in)

        db, settings, settings_in = parameters.convert_settings_orig(settings_in, db=db, settings=settings)

        db['data_path'] = data_path_all[j:j+1]
        db['save_path0'] = data_path_all[j]
        settings['classification']['preclassify'] = 0.0
        settings['run']['multiplane_parallel'] = 0
        settings['io']['delete_bin'] = True
        settings['detection']['sparsery_settings']['max_ROIs'] = 20000
        settings['run']['do_zcorr'] = False 

        run_s2p(db=db, settings=settings)


""" Track2p had to be edited to run mesoscope ROIs - add `plane_idx` as input to `get_all_roi_array_from_stat` function in loaders.py, called in loop.py. """

def run_t2p(db_exp, root='/media/carsen/disk2/CA1_tracking/'):
    from track2p.t2p import run_t2p                     # main function that launches track2p
    from track2p.ops.default import DefaultTrackOps     # default track2p options
    import numpy as np
    # load default settings / parameters
    track_ops = DefaultTrackOps()


    track_ops.all_ds_path = [f'/media/carsen/disk2/CA1_tracking/{exp["mname"]}/{exp["datexp"]}/{exp["blk"]}' for exp in db_exp]
            

    track_ops.save_path = f'/media/carsen/disk2/CA1_tracking/{db_exp[0]["mname"]}/'    # path where to save the outputs of algorithm 
                                                                                # (a 'track2p' folder will be created where figures for
                                                                                # visualisation and matrices of matches would be saved)

    track_ops.reg_chan = 0             # channel to use for registration (0=functional, 1=anatomical) (use 0 if only recording gcamp!)
    track_ops.iscell_thr = 0.5          # threshold for iscell (0.5 is a good value)
    track_ops.save_in_s2p_format = True

    ## takes 4-5 hours to run on 7-8 sessions
    run_t2p(track_ops)