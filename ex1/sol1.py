import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

GREYSCALE = 1
COLOR = 2
RGBDIM = 3
MIN_PIX_VAL = 0
MAX_PIX_VAL = 255

def is_valid_args(filename: str, representation: int) -> bool:
    """
    Basic checks on the functions input
    """
    return (filename is not None) and \
           (representation == 1 or representation == 2) and \
           isinstance(filename, str)


def is_rgb(im: np.ndarray) -> bool:
    """
    Check if a given image is rgb or greyscale
    """
    return im.ndim == RGBDIM


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

    if is_rgb(im) and (representation == GREYSCALE):  # change rgb to greyscale
        return rgb2gray(im).astype(np.float32)

    elif not is_rgb(im) and (representation == COLOR):
        raise Exception("Converting greyscale to RGB is not supported")

    return im.astype(np.float32) / MAX_PIX_VAL


def imdisplay(filename: str, representation: int) -> None:
    """
    display a given image with the given representation code
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining if the output should be either a
    greyscale image (1) or an RGB image (2)
    """

    plt.figure()
    try:
        if representation == GREYSCALE:
            plt.imshow(read_image(filename, representation), cmap=plt.cm.gray)
        else:
            plt.imshow(read_image(filename, representation))
    except:
        raise
    else:
        plt.show()


def rgb2yiq(imRGB: np.ndarray) -> np.ndarray:
    """
    transform an RGB image into the YIQ color space
    :param imRGB: height×width×3 np.float32 matrix with values in [0, 1]
    :return: YIQ color space
    """

    trans_mat = np.array([[0.299, 0.587, 0.114], [0.569, -0.275, -0.321], [0.212, -0.523, 0.311]]).astype(np.float32)
    if is_rgb(imRGB):
        return imRGB.dot(trans_mat.T)
    else:
        raise Exception("imRGB must be a RGB image")


def yiq2rgb(imYIQ: np.ndarray) -> np.ndarray:
    """
    transform an YIQ color space to RGB image
    :param imYIQ: height×width×3 np.float32 matrix with values in [0, 1]
    :return: RGB image
    """

    trans_mat = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.106, 1.703]]).astype(np.float32)
    if is_rgb(imYIQ):
        return imYIQ.dot(trans_mat.T)
    else:
        raise Exception("imYIQ must be an YIQ matrices")


def get_hist_cdf_and_yiq(im_orig: np.ndarray) -> tuple:
    """
    help function to get the cdf of an image
    :param im_orig: greyscale or RGB float32 image with values in [0, 1]
    :return: tuple contains: (orig_hist, bin_edges, cdf, yiq_mat)
    orig_hist - the original images' histogram, in case of a RGB image - the histogram of the Y matrix.
    bin_edges - the edges of the original images' histogram, in case of a RGB image - the histogram of the Y matrix.
    cdf - the cumulative histogram
    yiq_mat - the YIQ matrix in case of a RGB image, None otherwise
    """
    yiq_mat = None
    if is_rgb(im_orig):
        yiq_mat = rgb2yiq(im_orig)  # convert to YIQ and take only Y matrix
        im_orig = yiq_mat[:, :, 0]

    im_orig = (im_orig * MAX_PIX_VAL).round().astype(np.uint8)
    hist_orig, bin_edges = np.histogram(im_orig, MAX_PIX_VAL + 1, [MIN_PIX_VAL, MAX_PIX_VAL])
    cdf = np.cumsum(hist_orig)

    return hist_orig, bin_edges, cdf, yiq_mat


def histogram_equalize(im_orig: np.ndarray) -> tuple:
    """
    perform histogram equalization of a given greyscale or RGB image
    :param im_orig: greyscale or RGB float32 image with values in [0, 1]
    :return: [im_eq, hist_orig, hist_eq] -
    im_eq - is the equalized image. greyscale or RGB float32 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image.
    hist_eq - is a 256 bin histogram of the equalized image.
    """

    hist_orig, bin_edges, cdf, yiq_mat = get_hist_cdf_and_yiq(im_orig)
    norm_cdf = np.round(MAX_PIX_VAL * (cdf - min(cdf)) / (max(cdf) - min(cdf)))
    im_eq = np.interp(im_orig, bin_edges[:-1], norm_cdf).astype(np.float32) / MAX_PIX_VAL
    hist_eq, bin_edges_eq = np.histogram(im_eq * MAX_PIX_VAL, MAX_PIX_VAL + 1, [MIN_PIX_VAL, MAX_PIX_VAL])

    if yiq_mat is not None:  # im_eq needs to convert to RGB
        yiq_mat[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq_mat).clip(0, 1)

    return im_eq, hist_orig, hist_eq



def quantize(im_orig: np.ndarray, n_quant: int, n_iter: int) -> np.ndarray:

    hist_orig, bin_edges, cdf, yiq_mat = get_hist_cdf_and_yiq(im_orig)
    errors_arr = []

    # calc initial z
    z_arr = np.zeros(n_quant + 1, int)
    for i in range(1, n_quant):  # start from 1, first val is 0
        val = (i/n_quant)*max(cdf)
        z_arr[i] = np.searchsorted(cdf, val)
    z_arr[n_quant] = MAX_PIX_VAL  # last val is 255
    q_arr = np.zeros(n_quant, int)
    for it in range(n_iter):

        curr_err = 0
        # calc q and the error of the current iteration
        for i in range(n_quant):  # TODO check the borders, right now each z_i is calculates twice (except 0 and 255)
            z_min = z_arr[i]
            z_max = z_arr[i+1]
            q_arr[i] = np.average(hist_orig[z_min:z_max+1], weights=range(z_min, z_max+1))

            # calc error:
            curr_err += sum(hist_orig[z_min:z_max+1] * np.square(np.arange(z_min, z_max+1) - q_arr[i]))

        errors_arr.append(curr_err)

        # calc new z values, the borders (0 and 255) remains the same:
        new_z_arr = np.zeros(n_quant + 1, int)
        for i in range(1, n_quant):  # start from 1, first val is 0 #TODO solve bug: the new_z_arr gets [   0 1731 1915 1506  255] - needs to fix to the index of q!!
            new_z_arr[i] = (q_arr[i-1] + q_arr[i]) / 2
        new_z_arr[n_quant] = MAX_PIX_VAL  # last val is 255

        if False in (new_z_arr == z_arr):
            z_arr = new_z_arr.copy()
        else:  # got convergence!
            break


    # quantise the histogram
    for i in range(n_quant):
        hist_orig[z_arr[i]:z_arr[i+1]+1] = q_arr[i]

    im_quant = np.interp(im_orig, bin_edges[:-1], hist_orig).astype(np.float32) / MAX_PIX_VAL
    if yiq_mat is not None:  # im_eq needs to convert to RGB
        yiq_mat[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq_mat).clip(0, 1)

    return im_quant, errors_arr


res = quantize(read_image("tests/external/jerusalem.jpg", 1),4,5)
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(read_image("tests/external/jerusalem.jpg", 1), cmap=plt.cm.gray)
f.add_subplot(1, 2, 2)
plt.imshow(res[0], cmap=plt.cm.gray)
plt.show()





