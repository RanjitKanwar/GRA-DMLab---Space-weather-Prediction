import pytest
import numpy as np

from scipy.stats import skew, kurtosis, moment

from py_src.swdatatoolkit.imageparam import PatchSize
from py_src.swdatatoolkit.imageparam import MeanParamCalculator, EntropyParamCalculator, StdDeviationParamCalculator, \
    SkewnessParamCalculator, KurtosisParamCalculator, RelativeSmoothnessParamCalculator, UniformityParamCalculator, \
    TContrastParamCalculator, TDirectionalityParamCalculator, FractalDimParamCalculator
from py_src.swdatatoolkit.imageparam.util import PeakDetector, EdgeDetector
from unittest.mock import Mock
from py_src.swdatatoolkit.edgedetection import GradientCalculator
def test_patch_size_enum_import():
    assert PatchSize.FOUR != PatchSize.FULL


####################################
# Test MeanParamCalculator
####################################
def test_mean_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    with pytest.raises(TypeError):
        MeanParamCalculator(patch_size)


def test_mean_throws_on_null_data():
    patch_size = PatchSize.FOUR
    test_obj = MeanParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_mean_throws_on_wrong_data_type():
    data = [0, 1, 2]
    patch_size = PatchSize.FOUR
    test_obj = MeanParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_mean_throws_on_wrong_data_width():
    data = np.arange(8).reshape((4, 2))
    patch_size = PatchSize.FOUR
    test_obj = MeanParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_mean_throws_on_wrong_data_height():
    data = np.arange(8).reshape((2, 4))
    patch_size = PatchSize.FOUR
    test_obj = MeanParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_mean_full():
    data = [2, 2, 2, 2]
    data = np.reshape(np.array(data), (2, 2))
    patch_size = PatchSize.FULL
    test_obj = MeanParamCalculator(patch_size)
    val = test_obj.calculate_parameter(data)
    assert val == 2


def test_mean():
    data = np.arange(16).reshape((4, 4))
    patch_size = PatchSize.FOUR
    test_obj = MeanParamCalculator(patch_size)

    result = np.mean(data)
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result


############################################
# Test StdDeviationParamCalculator
############################################
def test_std_dev_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    with pytest.raises(TypeError):
        StdDeviationParamCalculator(patch_size)


def test_std_dev_throws_on_null_data():
    patch_size = PatchSize.FOUR
    test_obj = StdDeviationParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_std_dev_throws_on_wrong_data_type():
    data = [0, 1, 2]
    patch_size = PatchSize.FOUR
    test_obj = StdDeviationParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_std_dev_throws_on_wrong_data_width():
    data = np.arange(8).reshape((4, 2))
    patch_size = PatchSize.FOUR
    test_obj = StdDeviationParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_std_dev_throws_on_wrong_data_height():
    data = np.arange(8).reshape((2, 4))
    patch_size = PatchSize.FOUR
    test_obj = StdDeviationParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_std_dev_full():
    data = [2, 2, 2, 2]
    data = np.reshape(np.array(data), (2, 2))
    patch_size = PatchSize.FULL
    test_obj = StdDeviationParamCalculator(patch_size)
    val = test_obj.calculate_parameter(data)
    assert val == 0


def test_std_dev():
    data = np.arange(16).reshape((4, 4))
    patch_size = PatchSize.FOUR
    test_obj = StdDeviationParamCalculator(patch_size)

    result = np.std(data)
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result


########################################
# Test SkewnessParamCalculator
########################################
def test_skew_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    with pytest.raises(TypeError):
        SkewnessParamCalculator(patch_size)


def test_skew_throws_on_null_data():
    patch_size = PatchSize.FOUR
    test_obj = SkewnessParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_skew_throws_on_wrong_data_type():
    data = [0, 1, 2]
    patch_size = PatchSize.FOUR
    test_obj = SkewnessParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_skew_throws_on_wrong_data_width():
    data = np.arange(8).reshape((4, 2))
    patch_size = PatchSize.FOUR
    test_obj = SkewnessParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_skew_throws_on_wrong_data_height():
    data = np.arange(8).reshape((2, 4))
    patch_size = PatchSize.FOUR
    test_obj = SkewnessParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_skew_full():
    data = [2, 2, 2, 2]
    data = np.reshape(np.array(data), (2, 2))
    patch_size = PatchSize.FULL
    test_obj = SkewnessParamCalculator(patch_size)
    val = test_obj.calculate_parameter(data)
    assert val == 0


def test_skew():
    data = np.arange(16).reshape((4, 4))
    patch_size = PatchSize.FOUR
    test_obj = SkewnessParamCalculator(patch_size)

    result = skew(data, axis=None)
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result


########################################
# Test KurtosisParamCalculator
########################################
def test_kurtosis_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    with pytest.raises(TypeError):
        KurtosisParamCalculator(patch_size)


def test_kurtosis_throws_on_null_data():
    patch_size = PatchSize.FOUR
    test_obj = KurtosisParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_kurtosis_throws_on_wrong_data_type():
    data = [0, 1, 2]
    patch_size = PatchSize.FOUR
    test_obj = KurtosisParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_kurtosis_throws_on_wrong_data_width():
    data = np.arange(8).reshape((4, 2))
    patch_size = PatchSize.FOUR
    test_obj = KurtosisParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_kurtosis_throws_on_wrong_data_height():
    data = np.arange(8).reshape((2, 4))
    patch_size = PatchSize.FOUR
    test_obj = KurtosisParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_kurtosis_full():
    data = [2, 2, 2, 2]
    data = np.reshape(np.array(data), (2, 2))
    patch_size = PatchSize.FULL
    test_obj = KurtosisParamCalculator(patch_size)
    val = test_obj.calculate_parameter(data)
    assert val == -3.0


def test_kurtosis():
    data = np.arange(16).reshape((4, 4))
    patch_size = PatchSize.FOUR
    test_obj = KurtosisParamCalculator(patch_size)

    result = kurtosis(data, axis=None)
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result


########################################
# Test RelativeSmoothnessParamCalculator
########################################
def test_rel_smooth_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    with pytest.raises(TypeError):
        RelativeSmoothnessParamCalculator(patch_size)


def test_rel_smooth_throws_on_null_data():
    patch_size = PatchSize.FOUR
    test_obj = RelativeSmoothnessParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_rel_smooth_throws_on_wrong_data_type():
    data = [0, 1, 2]
    patch_size = PatchSize.FOUR
    test_obj = RelativeSmoothnessParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_rel_smooth_throws_on_wrong_data_width():
    data = np.arange(8).reshape((4, 2))
    patch_size = PatchSize.FOUR
    test_obj = RelativeSmoothnessParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_rel_smooth_throws_on_wrong_data_height():
    data = np.arange(8).reshape((2, 4))
    patch_size = PatchSize.FOUR
    test_obj = RelativeSmoothnessParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_rel_smooth_full():
    data = [2, 2, 2, 2]
    data = np.reshape(np.array(data), (2, 2))
    patch_size = PatchSize.FULL
    test_obj = RelativeSmoothnessParamCalculator(patch_size)
    val = test_obj.calculate_parameter(data)

    ans = np.var(data)
    ans = 1 - (1.0 / (1 + ans))
    assert val == ans


def test_rel_smooth():
    data = np.arange(16).reshape((4, 4))
    patch_size = PatchSize.FOUR
    test_obj = RelativeSmoothnessParamCalculator(patch_size)

    val = test_obj.calculate_parameter(data)

    ans = np.var(data)
    ans = 1 - (1.0 / (1 + ans))
    assert val[0, 0] == ans


#######################################
# Test EntropyParamCalculator
#######################################
def test_entropy_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    n_bins = 1
    min_val = 0
    max_val = 1
    with pytest.raises(TypeError):
        EntropyParamCalculator(patch_size, n_bins, min_val, max_val)


def test_entropy_calculator_throws_on_null_n_bins_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = None
    min_val = 0
    max_val = 1
    with pytest.raises(TypeError):
        EntropyParamCalculator(patch_size, n_bins, min_val, max_val)


def test_entropy_calculator_throws_on_null_min_val_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = None
    max_val = 1
    with pytest.raises(TypeError):
        EntropyParamCalculator(patch_size, n_bins, min_val, max_val)


def test_entropy_calculator_throws_on_null_max_val_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = None
    with pytest.raises(TypeError):
        EntropyParamCalculator(patch_size, n_bins, min_val, max_val)


def test_entropy_calculator_throws_on_max_val_less_than_min_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 3
    max_val = 1
    with pytest.raises(ValueError):
        EntropyParamCalculator(patch_size, n_bins, min_val, max_val)


def test_entropy_calculator_throws_on_null_data():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10
    test_obj = EntropyParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_entropy_calculator_throws_on_data_wrong_type():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10

    data = [1, 2, 3, 4]
    test_obj = EntropyParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_entropy_throws_on_wrong_data_width():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10

    data = np.arange(8).reshape((4, 2))
    test_obj = EntropyParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_entropy_throws_on_wrong_data_height():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10

    data = np.arange(8).reshape((2, 4))
    test_obj = EntropyParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_entropy():
    patch_size = PatchSize.FOUR
    n_bins = 4
    min_val = 0
    max_val = 16

    data = np.arange(16).reshape((4, 4))
    test_obj = EntropyParamCalculator(patch_size, n_bins, min_val, max_val)

    result = 2
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result


#######################################
# Test UniformityParamCalculator
#######################################
def test_uniformity_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    n_bins = 1
    min_val = 0
    max_val = 1
    with pytest.raises(TypeError):
        UniformityParamCalculator(patch_size, n_bins, min_val, max_val)


def test_uniformity_calculator_throws_on_null_n_bins_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = None
    min_val = 0
    max_val = 1
    with pytest.raises(TypeError):
        UniformityParamCalculator(patch_size, n_bins, min_val, max_val)


def test_uniformity_calculator_throws_on_null_min_val_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = None
    max_val = 1
    with pytest.raises(TypeError):
        UniformityParamCalculator(patch_size, n_bins, min_val, max_val)


def test_uniformity_calculator_throws_on_null_max_val_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = None
    with pytest.raises(TypeError):
        UniformityParamCalculator(patch_size, n_bins, min_val, max_val)


def test_uniformity_calculator_throws_on_max_val_less_than_min_in_constructor():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 3
    max_val = 1
    with pytest.raises(ValueError):
        UniformityParamCalculator(patch_size, n_bins, min_val, max_val)


def test_uniformity_calculator_throws_on_null_data():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10
    test_obj = UniformityParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_uniformity_calculator_throws_on_data_wrong_type():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10

    data = [1, 2, 3, 4]
    test_obj = UniformityParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_uniformity_throws_on_wrong_data_width():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10

    data = np.arange(8).reshape((4, 2))
    test_obj = UniformityParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_uniformity_throws_on_wrong_data_height():
    patch_size = PatchSize.FOUR
    n_bins = 5
    min_val = 0
    max_val = 10

    data = np.arange(8).reshape((2, 4))
    test_obj = UniformityParamCalculator(patch_size, n_bins, min_val, max_val)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_uniformity():
    patch_size = PatchSize.FOUR
    n_bins = 4
    min_val = 0
    max_val = 16

    data = np.arange(16).reshape((4, 4))
    test_obj = UniformityParamCalculator(patch_size, n_bins, min_val, max_val)

    result = 0.25
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result


########################################
# Test TContrastParamCalculator
########################################
def test_t_contrast_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    with pytest.raises(TypeError):
        TContrastParamCalculator(patch_size)


def test_t_contrast_throws_on_null_data():
    patch_size = PatchSize.FOUR
    test_obj = TContrastParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)


def test_t_contrast_throws_on_wrong_data_type():
    data = [0, 1, 2]
    patch_size = PatchSize.FOUR
    test_obj = TContrastParamCalculator(patch_size)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)


def test_t_contrast_throws_on_wrong_data_width():
    data = np.arange(8).reshape((4, 2))
    patch_size = PatchSize.FOUR
    test_obj = TContrastParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_t_contrast_throws_on_wrong_data_height():
    data = np.arange(8).reshape((2, 4))
    patch_size = PatchSize.FOUR
    test_obj = TContrastParamCalculator(patch_size)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_t_contrast_full():
    data = [2, 2, 2, 2]
    data = np.reshape(np.array(data), (2, 2))
    patch_size = PatchSize.FULL
    test_obj = TContrastParamCalculator(patch_size)
    val = test_obj.calculate_parameter(data)

    assert val == 0.0


def test_t_contrast():
    data = np.arange(16).reshape((4, 4))
    patch_size = PatchSize.FOUR
    test_obj = TContrastParamCalculator(patch_size)

    val = test_obj.calculate_parameter(data)

    std = np.std(data)
    kurt = moment(data, moment=4, axis=None)
    ans = np.power(std, 2.0) / np.power(kurt, 0.25)
    assert val[0, 0] == ans


########################################
# Test TDirectionalityCalculator
########################################
def test_t_directionality_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    gradient_calculator = GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1
    with pytest.raises(TypeError):
        TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)


def test_t_directionality_throws_on_null_gradient_calculator_in_constructor():
    patch_size = PatchSize.FOUR
    gradient_calculator = None
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1
    with pytest.raises(TypeError):
        TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)


def test_t_directionality_throws_on_null_peak_detector_in_constructor():
    patch_size = PatchSize.FOUR
    gradient_calculator = GradientCalculator('sobel')
    peak_detector = None
    quantization_level = 1
    with pytest.raises(TypeError):
        TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)

def test_t_directionality_throws_on_null_quantization_level_in_constructor():
    patch_size = PatchSize.FOUR
    gradient_calculator = GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = None
    with pytest.raises(TypeError):
        TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)

def test_t_directionality_throws_on_null_data():
    patch_size = PatchSize.FOUR
    gradient_calculator = GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1
    test_obj = TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)

def test_t_directionality_throws_on_wrong_data_type():
    patch_size = PatchSize.FOUR
    gradient_calculator =  GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1

    data = [1,2,3,4]
    test_obj = TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)

def test_t_directionality_throws_on_wrong_data_width():
    patch_size = PatchSize.FOUR
    gradient_calculator  = GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1

    data = np.arange(8).reshape((4, 2))
    test_obj = TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_t_directionality_throws_on_wrong_data_height():
    patch_size = PatchSize.FOUR
    gradient_calculator = GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1

    data = np.arange(8).reshape((2, 4))
    test_obj = TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)

def test_t_directionality():
    patch_size = PatchSize.FOUR
    gradient_calculator = GradientCalculator('sobel')
    peak_detector = PeakDetector(1, 25, 0, True)
    quantization_level = 1

    data = np.arange(16).reshape((4, 4))
    test_obj = TDirectionalityParamCalculator(patch_size, gradient_calculator, peak_detector, quantization_level)

    result = 0
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result



########################################
# Test FractalDimParamCalculator
########################################
def test_f_dim_calculator_throws_on_null_patch_in_constructor():
    patch_size = None
    binary_colors = 94
    edge_detector = EdgeDetector(2.5, 7.5, 2, 16, False)
    with pytest.raises(TypeError):
        FractalDimParamCalculator(patch_size, edge_detector)


def test_f_dim_calculator_throws_on_null_edge_detector_in_constructor():
    patch_size = PatchSize.FOUR
    edge_detector = None
    with pytest.raises(TypeError):
        FractalDimParamCalculator(patch_size, edge_detector)

def test_f_dim_calculator_throws_on_null_data():
    patch_size = PatchSize.FOUR
    edge_detector = Mock(spec=EdgeDetector)

    test_obj = FractalDimParamCalculator(patch_size, edge_detector)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(None)

def test_f_dim_calculator_throws_on_wrong_data_type():
    patch_size = PatchSize.FOUR
    edge_detector = 2

    data = [1,2,3,4]
    test_obj = FractalDimParamCalculator(patch_size, edge_detector)
    with pytest.raises(TypeError):
        test_obj.calculate_parameter(data)

def test_f_dim_calculator_throws_on_wrong_data_width():
    patch_size = PatchSize.FOUR
    edge_detector = 2

    data = np.arange(8).reshape((4, 2))
    test_obj = FractalDimParamCalculator(patch_size,  edge_detector)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)


def test_f_dim_calculator_throws_on_wrong_data_height():
    patch_size = PatchSize.FOUR
    edge_detector = 2

    data = np.arange(8).reshape((2, 4))
    test_obj = FractalDimParamCalculator(patch_size, edge_detector)
    with pytest.raises(ValueError):
        test_obj.calculate_parameter(data)

def test_f_dim_calculator():
    patch_size = PatchSize.FOUR
    edge_detector = EdgeDetector(2.5, 7.5, 2, 16, False)
    data = np.arange(16).reshape((4, 4))
    test_obj = FractalDimParamCalculator(patch_size,  edge_detector)

    result = 0
    val = test_obj.calculate_parameter(data)
    assert val[0, 0] == result