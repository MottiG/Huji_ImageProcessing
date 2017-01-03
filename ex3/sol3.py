import numpy as np
import typing
from scipy import signal as sp_signal
from scipy.ndimage import filters
from scipy.misc import imread
import matplotlib.pyplot as plt

PYR_IDX = 0  # the index of pyr in the tuple returned by build_gaussian_pyramid function
MIN_SIZE = 16  # minimum size of an image
SAMPLE_FACTOR = 2  # determine down/up sampling frequency - e.g. when reducing image take one of each 2 pixels


def gaussian_kernel(size: int) -> np.ndarray:
    """
    create a gaussian kernel
    :param size: the size of the gaussian kernel in each dimension (an odd integer)
    :return: gaussian kernel as np.ndarray contains np.float32
    """
    if not size % 2: # if size is even number
        raise Exception("kernel size must be odd number")

    base = np.ones((1, 2), np.float32)  # gaussian kernel base - [1 1] vector
    kernel = base
    for i in range(size-2):
        kernel = sp_signal.convolve(kernel, base)
    kernel /= np.sum(kernel)  # normalize kernel
    return kernel


def reduce(im: np.ndarray, blur_filter: np.ndarray) -> np.ndarray:
    """
    reduce an image by blurring and reducing image size by half
    :param im: image to reduce
    :param blur_filter: filter to use for blurring
    :return: the reduced image
    """
    reduced_im = filters.convolve(im, blur_filter, mode='mirror')
    reduced_im = filters.convolve(reduced_im, blur_filter.T, mode='mirror')
    reduced_im = reduced_im[::SAMPLE_FACTOR, ::SAMPLE_FACTOR]  # take any second element of im (if SAMPLE_FACTOR==2)
    return reduced_im


def expend(im: np.ndarray, blur_filter: np.ndarray) -> np.ndarray:
    """
    expend an image by doubling the image size and then blurring
    :param im: image to expend
    :param blur_filter: filter to use for blurring
    :return: the expended image
    """
    expended_im = np.zeros(([SAMPLE_FACTOR*dim for dim in im.shape]), dtype=np.float32)
    expended_im[1::SAMPLE_FACTOR, 1::SAMPLE_FACTOR] = im  # zero padding each odd index (if SAMPLE_FACTOR==2)
    expended_im = filters.convolve(expended_im, 2 * blur_filter, mode='mirror')
    expended_im = filters.convolve(expended_im, 2 * blur_filter.T, mode='mirror')
    return expended_im


def build_gaussian_pyramid(im: np.ndarray, max_levels: int, filter_size: int) -> (list, np.ndarray):
    """
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: tuple contains list of the pyramid levels and the filter used to construct the pyramid
    """
    pyr = [im]
    filter_vec = gaussian_kernel(filter_size)
    for level in range(1, max_levels):
        if min(pyr[-1].shape) <= MIN_SIZE:
            break
        pyr.append(reduce(pyr[-1], filter_vec))
    return pyr, filter_vec


def build_laplacian_pyramid(im: np.ndarray, max_levels: int, filter_size: int) -> (list, np.ndarray):
    """
    Construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: tuple contains list of the pyramid levels and the filter used to construct the pyramid
    """
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = [gauss_pyr[i] - expend(gauss_pyr[i+1], filter_vec) for i in range(len(gauss_pyr)-1)]
    pyr.append(gauss_pyr[-1])  # add G_n level as is
    return pyr, filter_vec


def laplacian_to_image(lpyr: list, filter_vec: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """
    :param lpyr: list of laplacian pyramid images
    :param filter_vec: the filter used to create the pyramid
    :param coeff: vector of coefficient numbers to multiply each level
    :return: the reconstructed image as np.ndarray
    """
    im = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr)-1, 0, -1):
        im = expend(im, filter_vec) + lpyr[i-1] * coeff[i-1]
    return im




