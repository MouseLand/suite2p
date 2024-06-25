Spike deconvolution
---------------------------

Our spike deconvolution in the pipeline is based on the OASIS algorithm
(see `OASIS paper`_). We run it with only a non-negativity constraint -
no L0/L1 constraints (see this `paper`_ for more details on why).

We first baseline the traces using the rolling max of the rolling min.
Here is an example of how the pipeline processes the traces (and how to
run your own data separately if you want):

::

   # compute deconvolution
   from suite2p.extraction import dcnv
   import numpy as np

   tau = 1.0 # timescale of indicator
   fs = 30.0 # sampling rate in Hz
   neucoeff = 0.7 # neuropil coefficient
   # for computing and subtracting baseline
   baseline = 'maximin' # take the running max of the running min after smoothing with gaussian
   sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
   win_baseline = 60.0 # in seconds, window in which to compute max/min filters

   ops = {'tau': tau, 'fs': fs, 'neucoeff': neucoeff,
          'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline}

   # load traces and subtract neuropil
   F = np.load('F.npy')
   Fneu = np.load('Fneu.npy')
   Fc = F - ops['neucoeff'] * Fneu

   # baseline operation
   Fc = dcnv.preprocess(
        F=Fc,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )

   # get spikes
   spks = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

.. _OASIS paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423
.. _paper: http://www.jneurosci.org/content/38/37/7976
