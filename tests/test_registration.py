import numpy as np

from suite2p.registration.utils import spatial_smooth


def test_spatial_smooth_has_not_regressed_during_refactor():
    frames = np.ones((2, 3, 3))
    smoothed = spatial_smooth(frames, 2)
    expected = np.array([
        [[1.  , 1.  , 0.5 ],
         [1.  , 1.  , 0.5 ],
         [0.5 , 0.5 , 0.25]],

        [[1.  , 1.  , 0.5 ],
         [1.  , 1.  , 0.5 ],
         [0.5 , 0.5 , 0.25]]], dtype=np.float32)
    assert np.allclose(smoothed, expected)