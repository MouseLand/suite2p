"""
Tests for suite2p.detection.anatomical
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from suite2p.detection import anatomical


def _normalize99(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return (img - lo) / (hi - lo)


def test_select_rois_highpass_spatial_applied_once(monkeypatch):
    """highpass_spatial should subtract the gaussian-smoothed image a single time."""
    captured = {}

    def fake_roi_detect(img, diameter=None, settings=None, device=None):
        captured["img"] = img.copy()
        masks = np.zeros(img.shape, "uint16")
        return masks, np.zeros((0, 2)), diameter[0], np.zeros(0)

    monkeypatch.setattr(anatomical, "roi_detect", fake_roi_detect)
    monkeypatch.setattr(anatomical, "normalize99", _normalize99, raising=False)

    rng = np.random.default_rng(0)
    max_proj = rng.random((64, 64)).astype("float32")
    mean_img = rng.random((64, 64)).astype("float32")
    diameter = [12., 12.]
    settings = {"img": "max_proj", "highpass_spatial": 2}

    anatomical.select_rois(mean_img, max_proj, settings, diameter=diameter)

    expected = np.clip(_normalize99(max_proj), 0, 1)
    expected -= gaussian_filter(expected, diameter[1] * settings["highpass_spatial"])
    np.testing.assert_allclose(captured["img"], expected, rtol=1e-5, atol=1e-6)
