import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

GREYSCALE = 1
COLOR = 2
RGBDIM = 3
RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                      [0.569, -0.275, -0.321],
                      [0.212, -0.523, 0.311]])
YIQ2RGB = np.array([[1, 0.956, 0.621],
                      [1, -0.272, -0.647],
                      [1, -1.106, 1.703]])


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
        im = imread(filename).astype(np.float32) / 255
    except OSError:
        raise Exception("Filename should be valid image filename")

    if is_rgb(im) and (representation == GREYSCALE):  # change rgb to greyscale
        im = rgb2gray(im).astype(np.float32)
    elif not is_rgb(im) and (representation == COLOR):
        raise Exception("Converting greyscale to RGB is not supported")

    return im


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
    if is_rgb(imRGB):
        return imRGB.dot(RGB2YIQ.T).astype(np.float32)
    else:
        raise Exception("imRGB must be a RGB image")


def yiq2rgb(imYIQ: np.ndarray) -> np.ndarray:
    """
    transform an YIQ color space to RGB image
    :param imYIQ: height×width×3 np.float32 matrix with values in [0, 1]
    :return: RGB image
    """
    if is_rgb(imYIQ):
        return imYIQ.dot(YIQ2RGB.T).astype(np.float32)
    else:
        raise Exception("imYIQ must be an YIQ matrices")


def histogram_equalize(im_orig: np.ndarray) -> tuple:
    """
    perform histogram equalization of a given greyscale or RGB image
    :param im_orig: greyscale or RGB float32 image with values in [0, 1]
    :return: [im_eq, hist_orig, hist_eq] -
    im_eq - is the equalized image. greyscale or RGB float32 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image.
    hist_eq - is a 256 bin histogram of the equalized image.
    """
    yiq_mat = None
    if is_rgb(im_orig):
        yiq_mat = rgb2yiq(im_orig) # convert to YIQ and take only Y matrix
        im_orig = yiq_mat[:, :, 0]
    hist_orig, bin_edges = np.histogram(im_orig * 255, 256)
    cdf = np.cumsum(hist_orig)
    hist_eq = np.around(255*(cdf - cdf.min()) / (cdf.max() - cdf.min()))
    im_eq = hist_eq[(im_orig * 255).astype(np.uint8)].astype(np.float32) / 255
    if yiq_mat is not None:  # im_eq needs to convert to RGB
        yiq_mat[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq_mat)
    return im_eq, hist_orig, hist_eq

