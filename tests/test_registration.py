import numpy as np
from suite2p.registration import bidiphase, utils


def test_spatial_smooth_has_not_regressed_during_refactor():
    frames = np.ones((2, 3, 3))
    smoothed = utils.spatial_smooth(frames, 2)
    expected = np.array([
        [[1.  , 1.  , 0.5 ],
         [1.  , 1.  , 0.5 ],
         [0.5 , 0.5 , 0.25]],

        [[1.  , 1.  , 0.5 ],
         [1.  , 1.  , 0.5 ],
         [0.5 , 0.5 , 0.25]]], dtype=np.float32)
    assert np.allclose(smoothed, expected)


def test_positive_bidiphase_shift_shifts_every_other_line():
    orig = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])
    expected = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6, 7]]
    ])

    shifted = orig.copy()
    bidiphase.shift(shifted, 2)
    assert np.allclose(shifted, expected)


def test_negative_bidiphase_shift_shifts_every_other_line():
    orig = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])
    expected = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])

    shifted = orig.copy()
    bidiphase.shift(shifted, -2)
    assert np.allclose(shifted, expected)