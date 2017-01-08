import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy import signal as sp_signal
from scipy.ndimage import filters

GREYSCALE, COLOR, RGBDIM = 1, 2, 3
MAX_PIX_VAL = 255
PYR_IDX = 0  # the index of pyr in the tuple returned by build_gaussian_pyramid function
MIN_SIZE = 16  # minimum size of an image
GAUSSIAN_BASE = np.ones((1, 2), np.float32)  # gaussian kernel base - [1 1] vector
SAMPLE_FACTOR = 2  # determine down/up sampling frequency - e.g. when reducing image take one of each 2 pixels


def is_valid_args(filename: str, representation: int) -> bool:
    """
    Basic checks on the functions input
    """
    return (filename is not None) and \
           (representation == 1 or representation == 2) and isinstance(filename, str)


def read_image(filename: str, representation: int) -> np.ndarray:
    """
    Reads a given image file and converts it into a given representation
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining if the output should be either a
    greyscale image (1) or an RGB image (2)
    :return: Image represented by a matrix of class np.float32, normalized to the range [0, 1].
    """

    if not is_valid_args(filename, representation):
        raise Exception("Please provide valid filename and representation code")

    try:
        im = imread(filename)
    except OSError:
        raise Exception("Filename should be valid image filename")

    if im.ndim == RGBDIM and (representation == GREYSCALE):  # change rgb to greyscale
        return rgb2gray(im).astype(np.float32)

    elif im.ndim != RGBDIM and (representation == COLOR):
        raise Exception("Converting greyscale to RGB is not supported")

    return im.astype(np.float32) / MAX_PIX_VAL


def gaussian_kernel_2d(size: int) -> np.ndarray:
    """
    create a gaussian kernel
    :param size: the size of the gaussian kernel in each dimension (an odd integer)
    :return: gaussian kernel as np.ndarray contains np.float32
    """
    if not size % 2: # if size is even number
        raise Exception("kernel size must be odd number")

    kernel = GAUSSIAN_BASE
    for i in range(size-2):
        kernel = sp_signal.convolve(kernel, GAUSSIAN_BASE)
    kernel = sp_signal.convolve2d(kernel, kernel.T,) / np.sum(kernel)  # change 1D to 2D kernel and normalize
    return kernel


def blur_spatial(im: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    perform image blurring using 2D convolution between the image im and a gaussian kernel g.
    :param im: the input image to be blurred (greyscale float32 image).
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer)
    :return: blurry image (greyscale float32 image).
    """
    if min(im.shape) < kernel_size: # if kernel_size smaller than min(dimensions) of the image
        raise Exception("kernel_size must be smaller or equal to the smallest dimension of the image")

    g = gaussian_kernel_2d(kernel_size)
    blur_im = sp_signal.convolve2d(im, g, 'same', 'wrap')  # using boundaries='wrap' so we will get periodic
    return blur_im



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
    doubled_filter = 2 * blur_filter
    expended_im = filters.convolve(expended_im, doubled_filter, mode='mirror')
    expended_im = filters.convolve(expended_im, doubled_filter.T, mode='mirror')
    return expended_im


def gaussian_kernel_1d(size: int) -> np.ndarray:
    """
    create a gaussian kernel
    :param size: the size of the gaussian kernel in each dimension (an odd integer)
    :return: gaussian kernel as np.ndarray contains np.float32
    """
    if not size % 2:  # if size is even number
        raise Exception("kernel size must be odd number")

    base = np.ones((1, 2), np.float32)  # gaussian kernel base - [1 1] vector
    kernel = base
    for i in range(size-2):
        kernel = sp_signal.convolve(kernel, base)
    kernel /= np.sum(kernel)  # normalize kernel
    return kernel


def build_gaussian_pyramid(im: np.ndarray, max_levels: int, filter_size: int) -> (list, np.ndarray):
    """
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: tuple contains list of the pyramid levels and the filter used to construct the pyramid
    """
    filter_vec = gaussian_kernel_1d(filter_size)
    pyr = [im]
    for lvl in range(1, max_levels):
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
    im = lpyr[-1]*coeff[-1]
    for i in range(len(lpyr)-1, 0, -1):
        im = expend(im, filter_vec) + lpyr[i-1]*coeff[i-1]
    return im


def pyramid_blending(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray,
                     max_levels: int, filter_size_im: int, filter_size_mask: int) -> np.ndarray:
    """
    pyramid blending as described in the lecture
    :param im1: first grayscale image to be blended
    :param im2: second grayscale image to be blended
    :param mask: boolean  mask representing which parts of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the maximal number of levels to use in the pyramids.
    :param filter_size_im: size of the Gaussian filter used in the construction of the pyramids of im1
    and im2.
    :param filter_size_mask: size of the Gaussian filter used in the construction of the pyramid of the mask.
    :return: the blended image as np.ndarray
    """
    if im1.shape != im2.shape != mask.shape:
        raise Exception("im1, im2 and mask must agree on dimensions")

    l1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[PYR_IDX]
    g_m = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)[PYR_IDX]
    l_out = [g_m[k]*l1[k] + (1 - g_m[k])*l2[k] for k in range(len(l1))]
    im_blend = laplacian_to_image(l_out, filter_vec, np.ones(len(l_out), np.float32)).clip(0, 1)
    return im_blend
