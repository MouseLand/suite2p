# Spike deconvolution

Our spike deconvolution in the pipeline is based on the OASIS algorithm
(see [OASIS paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423)). We run it with only a non-negativity constraint -
no L0/L1 constraints (see this [paper](http://www.jneurosci.org/content/38/37/7976) for more details on why).

We first baseline the traces using the rolling max of the rolling min,
computed with GPU-accelerated max/min pooling via PyTorch.
Here is an example of how the pipeline processes the traces (and how to
run your own data separately if you want):

```python
# compute deconvolution
from suite2p.extraction import dcnv
import numpy as np
import torch

tau = 1.0 # timescale of indicator in seconds
fs = 30.0 # sampling rate in Hz
neucoeff = 0.7 # neuropil coefficient
batch_size = 100 # number of neurons per batch
# for computing and subtracting baseline
baseline = 'maximin' # take the running max of the running min after smoothing with gaussian
sig_baseline = 10.0 # in frames, standard deviation of gaussian with which to smooth
win_baseline = 60.0 # in seconds, window in which to compute max/min filters
prctile_baseline = 8.0 # percentile of trace to use as baseline (only used if baseline='prctile')

# load traces and subtract neuropil
F = np.load('F.npy')
Fneu = np.load('Fneu.npy')
Fc = F - neucoeff * Fneu

# baseline operation
Fc = dcnv.preprocess(
     F=Fc,
     baseline=baseline,
     win_baseline=win_baseline,
     sig_baseline=sig_baseline,
     fs=fs,
     prctile_baseline=prctile_baseline,
     batch_size=batch_size,
     device=torch.device('cuda'),
 )

# get spikes
spks = dcnv.oasis(F=Fc, batch_size=batch_size, tau=tau, fs=fs)
```
