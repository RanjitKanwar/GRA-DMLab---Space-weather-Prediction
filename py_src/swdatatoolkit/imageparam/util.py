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
from typing import List, Dict, Tuple

import numpy
import numpy as np
import math
import numbers


class PeakDetector:
    """
    This class is a designed to find dominant peaks of a time series based on the settings provided by the user. For
    a peak to be considered dominant, three constraints can be set: a threshold on the frequency domain, a minimum
    peak-to-peak distance, and a maximum number of dominant peaks.

    The main task is done in the method :func:`~util.PeakDetection.find_peaks` which initially finds all the peaks
    (i.e., any data point whose value is larger than both of its previous and next neighbors), sorts them by their
    height, and then removes those which do not fall into the set constraints.

    """

    def __init__(self,  length : int, threshold: float, max_peaks: int = 0, is_percentile: bool = True):
        # Default value for peakwidth set as 1 for PeakDetector function under var length
        # Default value of 25 set for threshold set for PeakDetector function
        if length is None:
            self.length = 1
        else:
            self.threshold = threshold
        if threshold is None:
            self.threshold = 25
        else:
            self.threshold = threshold

        """
        Constructor for the class PeakDetection

        :param  - : The radius of the neighborhood of an accepted peak within which no other peaks is allowed. To
            relax this condition, set it to 1 or 0. The results would be similar, i.e., no peaks will be removed
            because of the neighboring constraints.
        :param threshold: On the frequency domain of the time series below which all the peaks will be ignored.
            To relax this condition, set it to the minimum frequency value of the given sequence. If used in conjunction
            with `is_percentile` values must be between 0 and 100.
        :param max_peaks: The maximum number of peaks to be found, from the highest to lowest. If `n=0`, all
            of the detected peaks will be returned.
        :param is_percentile: If `true`, then the value of `threshold` is interpreted as the percentile. Then the
            accepted values are doubles within [0,100]. If false, then the given `double` value will be used directly
            as the actual threshold on the frequency domain.
        :type  - : int
        :type threshold: float
        :type max_peaks: int
        :type is_percentile: bool
        :return:

        """
        self._length  =  length
        self._threshold = threshold
        self._max_peaks = max_peaks
        self._is_percentile = is_percentile

        if self._is_percentile:
            if not isinstance(self._threshold, numbers.Number) or self._threshold > 100 or self._threshold < 0:
                raise ValueError("Percentile must lie within the interval [0, 100]!")

        if not isinstance(self._length , numbers.Number) or self._length  < 0:
            raise ValueError("The values of ` - ` must be a numeric value and cannot be negative")

        if not isinstance(self._max_peaks, numbers.Number) or self._max_peaks < 0:
            raise ValueError("The values of `max_peaks` must be a numeric value and cannot be negative")

    def find_peaks(self, data: List[float]) -> List[int]:
        """
        This method finds the peaks of the given time series based on the constraints provided for the class

        The returned list is sorted by the height of the peaks, so the indices are not ordered.

        :param data: The time series that peaks are to be searched for in.
        :return: A list of index locations for the peaks meeting the criteria, sorted by height of the peak.
        """

        candidate_peaks = self._find_candidate_peaks(data)

        # Find the largest peaks and remove smaller peaks within the exclusion region of those peaks
        sorted_candidates = sorted(candidate_peaks, key=lambda item: item[1], reverse=True)

        removed_candidates = {}
        result_peaks = []
        for candidate_peak in sorted_candidates:
            if not candidate_peak[0] in removed_candidates:
                from_idx = candidate_peak[0] - self._length
                to_idx = candidate_peak[0] + self._length
                for idx in range(from_idx, to_idx + 1):
                    if idx in candidate_peaks and not idx == candidate_peak[0]:
                        val = candidate_peaks.pop(idx)
                        removed_candidates[idx] = val

                result_peaks.append(candidate_peak[0])

                # Check if limit has been reached
                if len(result_peaks) >= self._max_peaks:
                    break

        return result_peaks

    def _find_candidate_peaks(self, data: List[float]) -> List[Tuple[int, float]]:
        """
        Finds the position of any peaks on the time series, excluding those that are below the defined threshold.

        :param data: The time series that peaks are to be searched for in.
        :return: A list of tuples where peak positions within the original data that meet the defined threshold
            value is the first item and the height of the peak is the second.
        """
        mid = 1
        end = len(data)
        peaks = []

        threshold = 0.0
        if self._is_percentile:
            threshold = numpy.percentile(data, int(self._threshold))
        else:
            threshold = self._threshold

        # Adding points slightly more negative to the ends allows for the beginning and end to be possible peaks
        #data_cpy = [data[0] - 0.1] + data
       # data_cpy.append(data[-1] - 0.1)
        data_cpy = np.concatenate(([data[0] - 0.1], data, [data[-1] - 0.1]))

        while mid <= end:
            if data_cpy[mid - 1] < data_cpy[mid] and data_cpy[mid + 1] < data_cpy[mid] \
                    and data[mid - 1] > threshold:
                peaks.append((mid - 1, data[mid - 1]))
            mid += 1
        return peaks


class EdgeDetector:

    GAUSSIAN_CUT_OFF = 0.005
    MAGNITUDE_SCALE = 100
    MAGNITUDE_LIMIT = 1000
    MAGNITUDE_MAX = MAGNITUDE_SCALE * MAGNITUDE_LIMIT
    """
    This class provides a configurable implementation of the Canny edge detection
    algorithm. This classic algorithm has a number of shortcomings, but remains
    an effective tool in many scenarios.
    """
    def __init__(self, lowThreshold: float,highThreshold: float,gaussianKernelRadius: float,gaussianKernelWidth : int, contrastNormalized : bool):
        """
        Constructor for the class CannyEdgeDetector
      
        @param lowThreshold         The low threshold for hysteresis. Suitable values
                                    for this parameter must be determined
                                    experimentally for each application. It is
                                    nonsensical (though not prohibited) for this
                                    value to exceed the high threshold value.<br>
                                    <i>Suggested value: 2.5f</i>
        @param highThreshold        The high threshold for hysteresis. Suitable
                                    values for this parameter must be determined
                                    experimentally for each application. It is
                                    nonsensical (though not prohibited) for this
                                    value to be less than the low threshold
                                    value.<br>
                                    <i>Suggested value: 7.5f</i>
        @param gaussianKernelRadius The radius of the Gaussian convolution kernel
                                    used to smooth the source image prior to gradient
                                    calculation.<br>
                                    <i>Suggested value: 2f</i>
        @param gaussianKernel -   The number of pixels across which the Gaussian
                                    kernel is applied. This implementation will
                                    reduce the radius if the contribution of pixel
                                    values is deemed negligible, so this is actually
                                    a maximum radius.<br>
                                    <i>Suggested value: 16</i>
        @param contrastNormalized   Whether the luminance data extracted from the
                                    source image is normalized by linearizing its
                                    histogram prior to edge extraction. <i>Suggested
                                    value: false</i> <br>
                                    <b>note:</b> Keep
                                    <code>contrastNormalized = false</code> since
                                    this part is not implemented.
        """
        self._lowThreshold = lowThreshold
        self._highThreshold = highThreshold
        self._gaussianKernelRadius= gaussianKernelRadius
        self._gaussianKernelWidth = gaussianKernelWidth
        self._contrastNormalized = contrastNormalized



        if self._lowThreshold < 0:
            raise ValueError()

        if self._highThreshold < 0:
            raise ValueError()

        if self._gaussianKernelWidth   < 2:
            raise ValueError() 

    def get_edges(self, sourceImage : numpy.ndarray, colors: List[float]) -> numpy.ndarray:

        height = len(sourceImage)
        length  = len(sourceImage[0])
        picsize =  length * height

        data = numpy.empty(picsize, dtype=object)
        magnitude = numpy.empty(picsize, dtype=object)
        xConv = numpy.empty(picsize, dtype=object)
        yConv = numpy.empty(picsize, dtype=object)
        xGradient = numpy.empty(picsize, dtype=object)
        yGradient = numpy.empty(picsize, dtype=object)

        data = np.array(sourceImage).flatten()

        #if (self._contrastNormalized): as this has been set default as false will not be called
          #  normalizeContrast(picsize, data)

        self._computeGradients( length , height, data, yConv, xConv, xGradient, yGradient, magnitude)

        low = round(self._lowThreshold * self.MAGNITUDE_SCALE)
        high = round(self._highThreshold * self.MAGNITUDE_SCALE)

        self.performHysteresis(low, high, height,  length, data, magnitude)

        self.thresholdEdges(picsize, data, colors)

        ##result = MatrixUtil.convertTo2DArray(data,  length , height)
        result = np.reshape(data, (length , height))
        return result

        """
 NOTE: The elements of the method below (specifically the technique for
       non-maximal suppression and the technique for gradient computation) are
       derived from an implementation posted in the following forum (with the clear
       intent of others using the code):
       http://forum.java.sun.com/thread.jspa?threadID=546211&start=45&tstart=0 My
       code effectively mimics the algorithm exhibited above. Since I don't know the
       providence of the code that was posted it is a possibility (though I think a
       very remote one) that this code violates someone's intellectual property
       rights. If this concerns you feel free to contact me for an alternative,
       though less efficient, implementation.
       """

    def _computeGradients(self,  width : int, height: int, data: List[float], yConv: List[float], xConv: List[float], xGradient: List[float], yGradient: List[float],magnitude: List[float]) -> None:
        
        #generate the gaussian convolution masks
        kernel = [0.0 for i in range(self._gaussianKernelWidth)]
        diffKernel = [0.0 for i in range(self._gaussianKernelWidth)]

        for kwidth in range(self._gaussianKernelWidth):
            g1 = self.gaussian(kwidth, self._gaussianKernelRadius)
            if g1 <= self.GAUSSIAN_CUT_OFF and kwidth >= 2:
                break
            g2 = self.gaussian(kwidth - 0.5, self._gaussianKernelRadius)
            g3 = self.gaussian(kwidth + 0.5, self._gaussianKernelRadius)
            kernel[kwidth] = (g1 + g2 + g3) / 3 / (2 * math.pi * self._gaussianKernelRadius * self._gaussianKernelRadius)
            diffKernel[kwidth] = g3 - g2


        initX = kwidth-1
        maxX =  width - (kwidth - 1)
        initY = width * (kwidth - 1)
        maxY = width * (height - (kwidth - 1))

        ##perform convolution in x and y directions

        x_idx, y_idx = np.meshgrid(np.arange(initX, maxX), np.arange(initY, maxY, width), indexing="ij")
        ##used numpy.meshgrid to create a grid of indexes, and then use that grid to index into your data array.
        ##This will flatten the nested loops, and utilize the vectorized capabilities of numpy. This should be faster than using nested loops,
        # as the data is large.
        index = x_idx + y_idx
        sumX = data[index] * kernel[0]
        sumY = sumX
        for xOffset in range(1, kwidth):
            x_offset = xOffset * np.ones_like(x_idx)
            y_offset = width * xOffset * np.ones_like(y_idx)
            sumY += kernel[xOffset] * (data[index - y_offset] + data[index + y_offset])
            sumX += kernel[xOffset] * (data[index - x_offset] + data[index + x_offset])
        yConv[index] = sumY
        xConv[index] = sumX

        for x in range(initX, maxX):
            for y in range(initY, maxY, width):
                sum = 0
                index = x + y
                for i in range(1, kwidth):
                    sum += diffKernel[i] * (yConv[index - i] - yConv[index + i])

                xGradient[index] = sum

        for x in range(kwidth, width - kwidth):
            for y in range(initY, maxY, width):
                sum = 0.0
                index = x + y
                yOffset = width
                for i in range(1, kwidth):
                    sum += diffKernel[i] * (xConv[index - yOffset] - xConv[index + yOffset])
                    yOffset += width

                yGradient[index] = sum

        initX = kwidth
        maxX = width - kwidth
        initY = width * kwidth
        maxY = width * (height - kwidth)
        for x in range(initX, maxX):
            for y in range(initY, maxY, width):
                index = x + y
                indexN = index - width
                indexS = index + width
                indexW = index - 1
                indexE = index + 1
                indexNW = indexN - 1
                indexNE = indexN + 1
                indexSW = indexS - 1
                indexSE = indexS + 1

                xGrad = xGradient[index]
                yGrad = yGradient[index]
                gradMag = math.hypot(xGrad, yGrad)

                # perform non-maximal suppression
                nMag = math.hypot(xGradient[indexN], yGradient[indexN])
                sMag = math.hypot(xGradient[indexS], yGradient[indexS])
                wMag = math.hypot(xGradient[indexW], yGradient[indexW])
                eMag = math.hypot(xGradient[indexE], yGradient[indexE])
                neMag = math.hypot(xGradient[indexNE], yGradient[indexNE])
                seMag = math.hypot(xGradient[indexSE], yGradient[indexSE])
                swMag = math.hypot(xGradient[indexSW], yGradient[indexSW])
                nwMag = math.hypot(xGradient[indexNW], yGradient[indexNW])
                tmp = 0


                """
                An explanation of what's happening here, for those who want to understand the
                source: This performs the "non-maximal suppression" phase of the Canny edge
                detection in which we need to compare the gradient magnitude to that in the
                direction of the gradient; only if the value is a local maximum do we
                consider the point as an edge candidate.
                
                We need to break the comparison into a number of different cases depending on
                the gradient direction so that the appropriate values can be used. To avoid
                computing the gradient direction, we use two simple comparisons: first we
                check that the partial derivatives have the same sign (1) and then we check
                which is larger (2). As a consequence, we have reduced the problem to one of
                four identical cases that each test the central gradient magnitude against
                the values at two points with 'identical support'; what this means is that
                the geometry required to accurately interpolate the magnitude of gradient
                function at those points has an identical geometry (upto
                right-angled-rotation/reflection).
                
                When comparing the central gradient to the two interpolated values, we avoid
                performing any divisions by multiplying both sides of each inequality by the
                greater of the two partial derivatives. The common comparand is stored in a
                temporary variable (3) and reused in the mirror case (4).
                """

                if (xGrad * yGrad <= 0) and (abs(xGrad) >= abs(yGrad)):
                    tmp = abs(xGrad * gradMag)
                    if tmp >= abs(yGrad * neMag - (xGrad + yGrad) * eMag) and tmp > abs(
                            yGrad * swMag - (xGrad + yGrad) * wMag):
                        magnitude[index] = gradMag if gradMag >= MAGNITUDE_LIMIT else int(MAGNITUDE_SCALE * gradMag)
                    else:
                        magnitude[index] = 0
                elif (xGrad * yGrad <= 0) and (abs(xGrad) < abs(yGrad)):
                    tmp = abs(yGrad * gradMag)
                    if tmp >= abs(xGrad * neMag - (yGrad + xGrad) * nMag) and tmp > abs(
                            xGrad * swMag - (yGrad + xGrad) * sMag):
                        magnitude[index] = gradMag if gradMag >= MAGNITUDE_LIMIT else int(MAGNITUDE_SCALE * gradMag)
                    else:
                        magnitude[index] = 0
                elif (xGrad * yGrad > 0) and (abs(xGrad) >= abs(yGrad)):
                    tmp = abs(xGrad * gradMag)
                    if tmp >= abs(yGrad * seMag + (xGrad - yGrad) * eMag) and tmp > abs(
                            yGrad * nwMag + (xGrad - yGrad) * wMag):
                        magnitude[index] = gradMag if gradMag >= MAGNITUDE_LIMIT else int(MAGNITUDE_SCALE * gradMag)
                    else:
                        magnitude[index] = 0
                elif (xGrad * yGrad > 0) and (abs(xGrad) < abs(yGrad)):
                    tmp = abs(yGrad * gradMag)
                    if tmp >= abs(xGrad * seMag + (yGrad - xGrad) * sMag) and tmp > abs(
                            xGrad * nwMag + (yGrad - xGrad) * nMag):
                        magnitude[index] = gradMag if gradMag >= MAGNITUDE_LIMIT else int(MAGNITUDE_SCALE * gradMag)
                    else:
                        magnitude[index] = 0
        

        self._finalMagnitude = magnitude


    def gaussian(self, x, sigma):
        inv_sigma = 1 / (2 * sigma * sigma)
        return math.exp(-x * x * inv_sigma)


    def performHysteresis(self, low, high, height, width, data, magnitude):
        """
            NOTE: this implementation reuses the data array to store both
            luminance data from the image, and edge intensity from the
            processing.
            This is done for memory efficiency, other implementations may wish
            to separate these functions.
            """
        data = np.zeros((height, width), dtype=np.float64)
        magnitude_default = np.array(list(map(lambda x: x if x is not None else 0, magnitude))).reshape(data.shape)
        hysteresis_mask = (data == 0) & (magnitude_default >= high)

        #hysteresis_mask = (data == 0) & (magnitude >= high) old logic
        coordinates = np.transpose(np.nonzero(hysteresis_mask))
        for x, y in coordinates:
            follow(x, y, low, width, height, data, magnitude)

    def follow(self, x1, y1, i1, threshold, width, height, data, magnitude):
        x0 = max(x1 - 1, 0)
        x2 = min(x1 + 1, width - 1)
        y0 = max(y1 - 1, 0)
        y2 = min(y1 + 1, height - 1)

        data[y1, x1] = magnitude[y1, x1]
        hysteresis_mask = ((np.indices((height, width)) != (y1, x1)) & (data == 0) & (magnitude >= threshold))
        coordinates = np.transpose(np.nonzero(hysteresis_mask))
        for x, y in coordinates:
            follow(x, y, threshold, width, height, data, magnitude)

    """
    It replaces any zero values with <code>cols[0]</code>, and any non-zero
      positive values with <code>cols[1]</code>.
      
      @param picsize the size of the image in a 1D array.
      @param data    the background (i.e., <code>cols[0]</code>) and foreground
                     (i.e., <code>cols[1]</code>) color intensities used for
                     showing the detected edges.
    """

    def thresholdEdges(self, picsize, data, cols):
        background, foreground = cols
        for i in range(picsize):
            data[i] = foreground if data[i] > 0 else background


    def _normalizeContrast(picsize : int, data : List[float]) -> None:
        """
        With new assumptions, this is not easy to change. So, I am ignoring this for
        now, since previously we set contrastNormalized = false, and therefore we
        never called this function.
        """

    def getFinalMagnitude(self):
        return self.finalMagnitude
















