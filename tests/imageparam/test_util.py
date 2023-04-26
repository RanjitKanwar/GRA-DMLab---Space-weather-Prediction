import pytest

import numpy
import numpy as np

from py_src.swdatatoolkit.imageparam.util import PeakDetector


####################################
# Test PeakDetection
####################################
def test_peak_detection_throws_negative_width_in_constructor():
    width = -1
    threshold = 10.0
    max_peaks = 5
    is_percentile = False

    with pytest.raises(ValueError):
        PeakDetector(width, threshold, max_peaks, is_percentile)


def test_peak_detection_throws_nan_width_in_constructor():
    width = None
    threshold = 10.0
    max_peaks = 5
    is_percentile = False

    with pytest.raises(ValueError):
        PeakDetector(width, threshold, max_peaks, is_percentile)


def test_peak_detection_throws_negative_max_peak_in_constructor():
    width = 5
    threshold = 10.0
    max_peaks = -5
    is_percentile = False

    with pytest.raises(ValueError):
        PeakDetector(width, threshold, max_peaks, is_percentile)


def test_peak_detection_throws_nan_max_peak_in_constructor():
    width = 5
    threshold = 10.0
    max_peaks = None
    is_percentile = False

    with pytest.raises(ValueError):
        PeakDetector(width, threshold, max_peaks, is_percentile)


def test_peak_detection_throws_negative_threshold_with_percentile_in_constructor():
    width = 5
    threshold = -10.0
    max_peaks = 5
    is_percentile = True

    with pytest.raises(ValueError):
        PeakDetector(width, threshold, max_peaks, is_percentile)


def test_peak_detection_throws_nan_threshold_with_percentile_in_constructor():
    width = 5
    threshold = None
    max_peaks = 5
    is_percentile = True

    with pytest.raises(ValueError):
        PeakDetector(width, threshold, max_peaks, is_percentile)


def test_peak_detection_with_threshold():
    width = 2
    threshold = 1
    max_peaks = 5
    is_percentile = False

    data = [5, 5, 5, 9, 5, 5, 6, 8]
    expected = [3, 7]

    detector = PeakDetector(width, threshold, max_peaks, is_percentile)
    result = detector.find_peaks(data)
    assert result == expected


def test_peak_detection_with_threshold_and_limit():
    width = 2
    threshold = 1
    max_peaks = 1
    is_percentile = False

    data = [5, 5, 5, 9, 5, 5, 6, 8]
    expected = [3]

    detector = PeakDetector(width, threshold, max_peaks, is_percentile)
    result = detector.find_peaks(data)
    assert result == expected
