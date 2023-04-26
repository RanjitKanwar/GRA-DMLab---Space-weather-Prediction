"""
 swdatatoolkit, a project at the Data Mining Lab
 (http://dmlab.cs.gsu.edu/) of Georgia State University (http://www.gsu.edu/).

 Copyright (C) 2022 Georgia State University

 This program is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the Free Software
 Foundation version 3.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 this program. If not, see <http://www.gnu.org/licenses/>.
"""


import numpy
import numpy as np
from . import PatchSize
from ..edgedetection import GradientCalculator, Gradient
from .util import PeakDetector, EdgeDetector

from typing import Callable
from scipy.stats import skew, kurtosis, moment, linregress
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression


        

class BaseParamCalculator(metaclass=ABCMeta):
    """
    This is a base abstract class for calculating parameters over some patch of a 2D array.

    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        if patch_size is None:
            raise TypeError("PatchSize cannot be None in ParamCalculator constructor.")
        self._patch_size = patch_size

    @property
    @abstractmethod
    def calc_func(self) -> Callable:
        """
        This polymorphic property is designed to return the parameter calculation function to be applied to each
        patch of the input data.

        :return: :py:class:`typing.Callable` that is the parameter calculation function over a patch of a 2D array.
        """
        pass

    def calculate_parameter(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        This polymorphic method is designed to compute some image parameter. The parameters shall be calculated by
        iterating over a given 2D in a patch by patch manner, calculating the parameter for the pixel values within that
        patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image

        :return: either a :py:class:`numpy.ndarray` of the parameter value for each patch within the original input
            :py:class:`numpy.ndarray`, or a single value representing the parameter value of the entire input
            :py:class:`numpy.ndarray`.

        """
        if data is None or not isinstance(data, np.ndarray):
            raise TypeError("Data cannot be None and must be of type numpy.ndarray")
        if self._patch_size is None:
            raise TypeError("PatchSize cannot be None in calculator.")

        image_h = data.shape[0]
        image_w = data.shape[1]

        if self._patch_size is PatchSize.FULL:
            return self.calc_func(data)

        p_size = self._patch_size.value

        if image_w % p_size != 0:
            raise ValueError("Width of data must be divisible by given patch size!")
        if image_h % p_size != 0:
            raise ValueError("Height of data must be divisible by given patch size!")

        div_h = image_h // p_size
        div_w = image_w // p_size

        vals = np.zeros((int(div_h), int(div_w)))
        for row in range(div_h):
            for col in range(div_w):
                start_r = p_size * row
                end_r = start_r + p_size
                start_c = p_size * col
                end_c = start_c + p_size
                vals[row, col] = self.calc_func(data[start_r:end_r, start_c:end_c])

        return vals


class MeanParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the mean parameter over some patch of a 2D array.
    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """

        super().__init__(patch_size)
        '''
        :return: The mean value for the patch passed in
        '''
        self._calc_func = np.mean

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


class StdDeviationParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the standard deviation parameter over some patch of a 2D array.

    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)
        '''
        :return: The standard deviation value for the patch passed in
        '''
        self._calc_func = np.std

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# SkewnessParamCalculator
####################################
class SkewnessParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the skewness parameter over some patch of a 2D array.

    See :py:class:`scipy.stats.skew` for additional information on the calculation for
    each cell.
    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        '''
        Using scipy.stats library skew to calculate skewness of the image from the patch image
        :param data: :py:class:`numpy.ndarray`
        2D matrix representing some image
        :return: The skewness parameter value for the patch passed in
        '''
        val = skew(data, axis=None)
        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# KurtosisParamCalculator
###################################
class KurtosisParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the kurtosis parameter over some patch of a 2D array.

    See :py:class:`scipy.stats.kurtosis` for additional information on the calculation for
    each cell.
    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        '''
        Using scipy.stats library kurtosis to calculate skewness of the image from the patch image
        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The kurtosis value for the patch passed in
        '''
        val = kurtosis(data, axis=None)
        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# RelativeSmoothnessParamCalculator
###################################
class RelativeSmoothnessParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the relative smoothness parameter over some patch of a 2D array.


    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        '''
        Compute the variance of the above image patch,by using the variance we'll calculate the smoothness
        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The smoothness parameter value for the patch passed in
        '''
        val = np.var(data)
        val = 1 - (1.0 / (1 + val))
        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# UniformityParamCalculator
###################################
class UniformityParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the uniformity parameter over some patch of a 2D array.


    """

    def __init__(self, patch_size: PatchSize, n_bins: int, min_val: float, max_val: float):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            The patch size to calculate the parameter over.
        :param n_bins: int or sequence of scalars or str
            The number of bins to use when constructing the frequency histogram for each patch.
            If bins is an int, it defines the number of equal-width bins in the given range.
            If bins is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths. See :py:class:`numpy.histogram`
            as it is used internally
        :param min_val: py:float
            The minimum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored
        :param max_val: float
            The maximum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored. The max_val must be greater than or equal to min_val.

        """
        super().__init__(patch_size)

        if n_bins is None:
            raise TypeError("n_bins cannot be None in UniformityParamCalculator constructor.")
        if min_val is None:
            raise TypeError("min_val cannot be None in UniformityParamCalculator constructor.")
        if max_val is None:
            raise TypeError("max_val cannot be None in UniformityParamCalculator constructor.")

        if min_val > max_val:
            raise ValueError("max_val cannot be less than min_val in UniformityParamCalculator constructor.")

        self._n_bins = n_bins
        self._range = (min_val, max_val)

    @property
    def calc_func(self) -> Callable:
        return self.__calc_uniformity

    def __calc_uniformity(self, data: numpy.ndarray) -> float:
        """
        Helper method that performs the uniformity calculation for one patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The uniformity parameter value for the patch passed in

        """

        hist, bin_edges = np.histogram(data, self._n_bins, range=self._range)
        image_h = data.shape[0]
        image_w = data.shape[1]
        
        '''
        Iterate over the normalized histogram of thisPatch (0 to nOfBins)
                Calculate uniformity for thisPatch:
                uniformity = SUM(p ^2)
        '''
        n_pix = float(image_w * image_h)
        sum = 0.0
        for i in range(len(hist)):
            count = hist[i]
            if count == 0:
                continue
            prob = hist[i] / n_pix
            sum += np.power(prob, 2)

        return sum


###################################
# EntropyParamCalculator
###################################
class EntropyParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the entropy parameter over some patch of a 2D array.
    The calculation is performed in the following manner:

    .. math:: E = - \\sum_{i=1}^{N} p(z_i)* log_2(p(z_i))

    where:

    - :math:`p` is the histogram of a patch

    - :math:`z_i` is the intensity value of the i-th pixel in the patch

    - :math:`p(z_i)` is the frequency of the intensity :math:`z_i` in the histogram of the patch

    """

    def __init__(self, patch_size: PatchSize, n_bins: int, min_val: float, max_val: float):
        # Default value of 12 bins set for entropy function
        if n_bins is None:
            self.n_bins = 12
        else:
            self.n_bins = n_bins
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            The patch size to calculate the parameter over.
        :param n_bins: int or sequence of scalars or str
            The number of bins to use when constructing the frequency histogram for each patch.
            If bins is an int, it defines the number of equal-width bins in the given range.
            If bins is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths. See :py:class:`numpy.histogram`
            as it is used internally
        :param min_val: py:float
            The minimum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored
        :param max_val: float
            The maximum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored. The max_val must be greater than or equal to min_val.

        """
        super().__init__(patch_size)

        if n_bins is None:
            raise TypeError("n_bins cannot be None in EntropyParamCalculator constructor.")
        if min_val is None:
            raise TypeError("min_val cannot be None in EntropyParamCalculator constructor.")
        if max_val is None:
            raise TypeError("max_val cannot be None in EntropyParamCalculator constructor.")

        if min_val > max_val:
            raise ValueError("max_val cannot be less than min_val in EntropyParamCalculator constructor.")

        self._n_bins = n_bins
        self._range = (min_val, max_val)

    @property
    def calc_func(self) -> Callable:
        return self.__calc_entropy

    def __calc_entropy(self, data: numpy.ndarray) -> float:
        """
        Helper method that performs the entropy calculation for one patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The entropy parameter value for the patch passed in

        """

        hist, bin_edges = np.histogram(data, self._n_bins, range=self._range)
        image_h = data.shape[0]
        image_w = data.shape[1]

        n_pix = float(image_w * image_h)
        sum = 0.0
        '''
        Iterate over the histogram of flatPatch (0 to nOfBins)
             Entropy = - SUM {p(z_i) * log_2(p(z_i))}
        '''
        for i in range(len(hist)):
            count = hist[i]
            if count == 0:
                continue
            prob = hist[i] / n_pix
            sum += prob * (np.log2(prob))

        return 0 - sum


###################################
# TContrastParamCalculator
###################################
class TContrastParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the Tamura Contrast parameter over some patch of a 2D array.
    The calculation is performed in the following manner:

    .. math:: C = \\frac{\\sigma^{2}}{{\\mu_4}^{0.25}}

    where:

    - :math:`\\sigma^{2}` is the variance of the intensity values in the patch

    - :math:`\\mu_4` is kurtosis (4-th moment about the mean) of the intensity values in the patch

    This formula is an approximation proposed by Tamura et al. in "Textual Features Corresponding Visual
    Perception" and investigated in "On Using SIFT Descriptors for Image Parameter Evaluation"

    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: Tthe patch size to calculate the parameter over.
        :type patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        '''
         :return: The Tcontrast parameter value for the patch passed in
        '''
        kurt_val = moment(data, moment=4, axis=None)
        if kurt_val == 0:
            return 0.0
        std_val = np.std(data)

        # TContrast = (sd ^ 2)/(kurtosis ^ 0.25)
        val = np.power(std_val, 2) / np.power(kurt_val, 0.25)
        if np.isnan(val):
            return 0.0

        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# TDirectionalityParamCalculator
###################################
class TDirectionalityParamCalculator(BaseParamCalculator):
#def main():
    def __init__(self, patch_size: PatchSize, gradient_calculator: GradientCalculator, peak_detector: PeakDetector,
                 quantization_level: int):
        """
        Constructor

        :param patch_size: The patch size to calculate the parameter over.
        :param gradient_calculator: Calculator for gradient of pixel values in the image being processed.
        :param peak_detector: Object that finds the local maxima in an ordered series of values
        :param quantization_level: The quantization level for the continuous spectrum of the angles of gradients. This
            is the number of bins in a histogram of gradient angles.
        :type patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
        """
        super().__init__(patch_size)

        if gradient_calculator is None:
            raise TypeError("gradient_calculator cannot be None in TDirectionalityParamCalculator constructor.")
        if peak_detector is None:
            raise TypeError("peak_detector cannot be None in TDirectionalityParamCalculator constructor.")
        if quantization_level is None:
            raise TypeError("quantization_level cannot be None in TDirectionalityParamCalculator constructor.")

        self._gradient_calculator = gradient_calculator
        self._peak_detector = peak_detector
        self._quantization_level = quantization_level
        self._radius_threshold_percentage = 0.15
        self._insignificant_radius = 1e-4

    @property
    def calc_func(self) -> Callable:
        return self.__calc_directionality
        


    def _find_middle_points(self, startingPoint, array, endingPoint):
        points = [0.0] * (len(array) + 1)
        firstIndex = 0
        lastIndex = len(points) - 1

        if (len(array) == 0):
            raise ValueError("The given collection of integers cannot be empty!")
        if (startingPoint > array[0] or endingPoint < array[len(array) - 1]):
            raise ValueError("Either StartingPoint or EndingPoint is not correct!")

        # first point is 'startingPoint'
        points[firstIndex] = startingPoint

        for i in range(1, len(points) - 1):
            points[i] = (array[i - 1] + array[i]) / 2

        # last point is 'endingPoint'
        points[lastIndex] = endingPoint

        return points

    def incremental_array(self,first, last):
        if first >= last:
            return None
        results = [first + i for i in range(last - first + 1)]
        return results

    def __calc_directionality(self, data: numpy.ndarray) -> float:
        '''
        :param data: :py:class:`numpy.ndarray`
        2D matrix representing some image
        :return: The TDirectionality parameter value for the patch passed in
        '''

        gradient = self._gradient_calculator.calculate_gradient_polar(data)

        gradient_theta = gradient.gx
        gradient_radii = gradient.gy
        
        radiusThreshold = self._radius_threshold_percentage * np.max(gradient_radii)
        
        for i in range(len(gradient_radii)):
            if (gradient_radii < radiusThreshold).all():
                gradient_radii[i] = 0
                
        for i in range(len(gradient_theta)):
            if (np.absolute(gradient_theta) < self._insignificant_radius).all():
                gradient_theta[i] = 0

        breaks = np.linspace(0, np.pi, num=self._quantization_level+ 1, endpoint=True)
        breaks = np.array(breaks, dtype=np.float64)

        temp_hist=np.histogram(gradient_theta,bins=self._quantization_level,range=(0,np.pi))
        hist_t = temp_hist[1]
        hist_t[0] = (hist_t[0] / 100)
        
        peaksIndex = self._peak_detector.find_peaks(hist_t)

        peaksIndex.sort()
        numberOfPeaks = len(peaksIndex)
        
        if (numberOfPeaks == 0):
            fDir = 0
            return fDir
        
        
        middlePoints = self._find_middle_points(0, peaksIndex, len(breaks) - 1)

        for i in range(0, numberOfPeaks, 1):
            ## 1. Get the interval
            _from = middlePoints[i]
            to =  middlePoints[i + 1]
            ## 2. Create the incremental array for this interval
            inc = self.incremental_array(_from, to)
            ## 3. Using the incremental array and the index of peak, compute the weights
            
            thisPeakIndex = peaksIndex[i]

            innerSum = 0
            outerSum = 0

            for j in range(0, len(inc) - 1, 1):
                innerSum += np.power(breaks[inc[j]] - breaks[thisPeakIndex], 2) * hist_t[inc[j]] 
            
            outerSum += innerSum
            
        fDir = numberOfPeaks * outerSum

        
        normVal = np.sum(hist_t)
        fDir = fDir / normVal
        
        return fDir
        



###################################
# FractalDimParamCalculator
###################################
class FractalDimParamCalculator(BaseParamCalculator):

    '''
    This class is designed to compute the <b>Fractal Dimension</b> parameter for
    each patch of the given <code>BufferedImage</code>. In this class, a <i>Box
    counting</i> approach known as <i>Minkowski�Bouligand Dimension</i> is
    implemented.
    '''

    def __init__(self, patch_size: PatchSize, edge_detector: EdgeDetector ):
        '''
        Constructor
    
        :param patchSize: the size of the boxes by which the image will be processed.
        :param edgeDetector: the algorithm to be used to extract the binary image of edges from the input image.
        '''
    
        super().__init__(patch_size)

        if (edge_detector is None):
            raise TypeError("edge_detector cannot be null in FractalDimParamcalculator \nYou need to provide a valid edge detector, or use the other contructor.")

        self._edge_detector = edge_detector
        self._patch_size = patch_size
        self._epsilon = 1e-32

        '''
        To calculate box sizes we first calculate maxbox size based on the PatchSize.Enum based on the formula return 2 * round(np.sqrt(pSize)) if its 0, 
        then fractal dimension is going to be calculated for the entire image at once. So, in this case, we set the maxBoxSize to 64 pixel.
        
        We then calculate the splits using the max box size using the formula         
        
        results = [np.log2(a * 1.0)]
        results[0] = a
        i = 1
        while (a > 1):
            i+=1
            results[i] = a
            a = a / 2
            
        '''

        if self._patch_size == PatchSize.ONE:
            self._boxSizes = [2.0, 1.0]
        elif self._patch_size == PatchSize.FOUR or self._patch_size == PatchSize.SIXTEEN:
            self._boxSizes = [4.0, 2.0, 1.0]
        elif self._patch_size == PatchSize.THIRTY_TWO:
            self._boxSizes = [16.0, 8.0, 4.0, 2.0, 1.0]
        elif self._patch_size in [PatchSize.SIXTY_FOUR, PatchSize.ONE_TWENTY_EIGHT, PatchSize.TWO_FIFTY_SIX]:
            self._boxSizes = [32.0, 16.0, 8.0, 4.0, 2.0, 1.0]
        elif self._patch_size in [None, PatchSize.FIVE_TWELVE, PatchSize.TEN_TWENTY_FOUR, PatchSize.FULL]:
            self._boxSizes = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]

        #self._maxBoxSize = pSize
        #self._boxSizes = self.getAllSplits(pSize)
    
    @property
    def calc_func(self) -> Callable:
        return self.__calc_fractal

    def __calc_fractal(self, data: numpy.ndarray) -> float:
        
    
        """
        Helper method that performs the fractal dim calculation for one patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The fractal dim parameter value for the patch passed in

        """
        if (self._edge_detector == None):
            raise ValueError("EdgeDetector cannot be null in FractalDimParamcalculator \nYou need to provide a valid edge detector, or use the other contructor.")
       # if (self._binary_colors == None):
           # raise ValueError("binaryColors cannot be null in FractalDimParamcalculator")
        
        image_h = data.shape[0]
        image_w = data.shape[1]
       
        """
        This method prepares the image for the box-counting method. It applies the
        given Edge Detection algorithm on the image (assigns white:255 to pixels
        representing the edges and black:0 to others), and then converts it into a 1D
        array and passes it to countBox method. If <code>null</code> was passed to
        the constructor of this class, then the given image will be considered
        binary, hence no preparation will be carried out.
     
        @param image
        @return a single double number as the Fractal Dimension of the given image.
        """
        counts = [0] * len(self._boxSizes)
        
        """If no edge detector is provided, then use the given colors"""
        if self._edge_detector == None:
            colors = self._binary_colors
            binaryImage = data
        else:
            colors = [0, 255] ## colors[0]: (B) background, colors[1]:(W) foreground
            binaryImage = self._edge_detector.get_edges(data, colors)
        

        g = np.array(binaryImage).flatten().astype(np.float64)
        """Do counting for boxes of different sizes."""
        for i in range(0, len(self._boxSizes), 1):
            counts[i] = self._countBoxes(g, image_w, image_h, self._boxSizes[i], colors)


        """
        Find the regression slope for the points in the plot X: log(boxSize), Y:
        * log(counts)
        """
        #for i in range(0, len(self._boxSizes)-1, 1):
            #slope = slope + np.polyfit(np.log(self.boxSizes[i]), np.log(counts[i]),1)[0]


        reg = LinearRegression()

        for i in range(len(counts)):
            reg.fit(np.log(self._boxSizes[i]).reshape(-1, 1), np.log(counts[i]).reshape(-1, 1))

        slope = reg.coef_[0][0] if np.isfinite(reg.coef_[0][0]) else 0

        """
        The regression slope of such data is always negative, but we only care about
        the magnitude of the slope, hence the absolute value.
        """
        return abs(slope)

    '''
    NO LONGER IN USE
    def findMaxBoxSize(pSize : int) -> int: 

        
        if pSize=0, then fractal dimension is going to be calculated for the entire
         image at once. So, in this case, we set the maxBoxSize to 64 pixel.
        
        if pSize == 0:
            return 64
        elif pSize == 1:
            return 2
        elif pSize == 4:
            return 4
        elif pSize == 16:
            return 4
        elif pSize == 32:
            return 16
        elif pSize == 64:
            return 32
        elif pSize == 128:
            return 32
        elif pSize == 256:
            return 32
        elif pSize == 512:
            return 64
        elif pSize == 1024:
            return 64
        elif pSize == -1:
            return 64

        #return 2 * round(np.sqrt(pSize))

    def getAllSplits(a : int) -> int:

        if a < 2:
            raise ValueError("The argument is too small to be split.")


        results = [np.log2(a * 1.0)]
        results[0] = a
        i = 1
        while (a > 1):
            i+=1
            results[i] = a
            a = a / 2
        return results
    
    '''

    '''def _countBoxes(self,data : numpy.ndarray, imageW : int, imageH : int, boxSize : int,colors : numpy.ndarray) -> int:
        if data.shape != (imageH, imageW):
            raise TypeError("the given array does not match with the given width and height!")
        if (boxSize > imageW) or (boxSize > imageH):
            raise ValueError("The give boxSize is larger than the patch on which the box counting should take place")
    
        subPatch = []
        boxW = boxSize
        boxH = boxSize
        done = 0
        x=0
        y=0
        boxCounter=0

        while done is False:
            subPatch = self.getSubMatrix(data, imageW, imageH, x, y, boxW, boxH)
            
            for i in range (0, len(subPatch)-1, 1):
                if subPatch[i] == colors[1]: 
                    """
                     If subPatch has any foreground color in it
                     this subPatch spans over a segment of an edge, so it should be counted and
                     there is no need to proceed any further.
                    """
                    boxCounter+=1
                    break
                    
            x += boxSize
            if x + boxSize > imageW:
                 """If the remaining horizontal space is less than a box shrink the box horizontally to fit the remaining space """
                 boxW = imageW % boxSize
                 
                 if x >= imageW:
                    #Reset w
                    boxW = boxSize
                    #Reset x
                    x = 0
                    #shift y
                    y += boxSize
                    if y + boxSize > imageH: 
                        """
                        If the remaining vertical space is less than a box shrink the box vertically to 
                        fit the remaining space done if the entire image is covered
                        """
                        boxH = imageH % boxSize
                        
                        done = (y >= imageH)
                        
        return boxCounter
        '''

    import numpy as np

    def _countBoxes(self, image, imageW, imageH, boxSize, colors):
        if len(image) != imageW * imageH:
            raise ValueError("the given array does not match with the given width and height!")
        if boxSize > imageW or boxSize > imageH:
            raise ValueError("The give boxSize is larger than the patch on which the box counting should take place")

        subPatch = []
        x, y = 0, 0
        boxW, boxH = boxSize, boxSize
        done = False
        boxCounter = 0

        while not done:
            subPatch = self.getSubMatrix(image, imageW, imageH, x, y, boxW, boxH)

            for i in range(len(subPatch)):
                if subPatch[i] == colors[1]:
                    # this subPatch spans over a segment of an edge, so it should be counted and
                    # there is no need to proceed any further.
                    boxCounter += 1
                    break

            # Move the box
            x += boxSize
            if x + boxSize > imageW:
                # If the remaining horizontal space is less than a box
                boxW = imageW % boxSize  # shrink the box horizontally to fit the remaining space
                if x >= imageW:
                    # Reset w
                    boxW = boxSize
                    # Reset x
                    x = 0
                    # shift y
                    y += boxSize
                    if y + boxSize > imageH:
                        # If the remaining vertical space is less than a box
                        boxH = imageH % boxSize  # shrink the box vertically to fit the remaining space
                        if y >= imageH:
                            done = True  # done if the entire image is covered

        return boxCounter

        """
       This method simply returns a sub-matrix of a given matrix, except that its
       input and output are both 1D arrays representing 2D matrices.
       
       @param matrix
                  The given matrix whose sub-matrix is inquired. This is a 1D array
                  representing a matrix.
       @param rowLength
                  The length of each row of the given matrix.
       @param colLength
                  The length of each column of the given matrix.
       @param xBox
                  The x coordinate of the inquired sub-matrix on the given matrix.
       @param yBox
                  The y coordinate of the inquired sub-matrix on the given matrix.
       @param boxW
                  The length of each row of the inquired sub-matrix.
       @param boxH
                  The length of each column of the inquired sub-matrix.
       @return a 1D array representing the inquired sub-matrix of the given matrix.
      """


    '''
    def getSubMatrix(self, data : numpy.ndarray,rowLength : int ,colLength : int, xBox : int, yBox : int, boxW : int, boxH : int ) -> list[float]:
        
        if data.length != (rowLength * colLength):
            raise ValueError("the given matrix doesn't match with the given rowLength and colLength!")

        if boxW > rowLength or boxH > colLength:
            raise ValueError("The expected sub-matrix is bigger than the given matrix!")

        if boxW + xBox > rowLength or boxH + yBox > colLength:
            raise ValueError("The expected sub-matrix is out of the boundary of the given matrix!")
            
            
        subMatrix = [boxW * boxH]
        k=0
        
        for i in range(0, boxH - 1, 1):
            for j in range(0, boxW - 1, 1):
                subMatrix[k] = data[(xBox + j) + ((yBox + i) * rowLength)]
                k+=1

        return subMatrix
        '''

    def getSubMatrix(self, data : numpy.ndarray, rowLength, colLength, xBox, yBox, boxW, boxH):
        if len(data) != rowLength * colLength:
            raise ValueError("the given matrix doesn't match with the given rowLength and colLength!")

        if boxW > rowLength or boxH > colLength:
            raise ValueError("The expected sub-matrix is bigger than the given matrix!")

        if boxW + xBox > rowLength or boxH + yBox > colLength:
            raise ValueError("The expected sub-matrix is out of the boundary of the given matrix!")

        subMatrix = []
        boxH_int=int(boxH)
        boxW_int = int(boxW)
        xBox_int = int(xBox)
        yBox_int = int(yBox)
        for i in range(boxH_int):
            for j in range(boxW_int):
                subMatrix.append(data[(xBox_int + j) + ((yBox_int + i) * rowLength)])

        return subMatrix



        

                 
             
     
     
        
        
        
        
        
        
                
        
                
        


