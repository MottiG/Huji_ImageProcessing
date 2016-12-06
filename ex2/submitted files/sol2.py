import numpy as np
from scipy import signal as sp_signal

STANDARD = 1
CONJUGATE = -1
GAUSSIAN_BASE = np.ones((1, 2), np.float32)  # gaussian kernel base - [1 1] vector


# # # # # # # # # # START OF COPY&PASTE FROM EX1 - THE PRESUBMIT USES THE READ_IMAGE FUNCTION # # # # # # # # # #
from scipy.misc import imread
from skimage.color import rgb2gray
GREYSCALE, COLOR, RGBDIM = 1, 2, 3
MAX_PIX_VAL = 255
def is_valid_args(filename: str, representation: int) -> bool:
    return (filename is not None) and (representation == 1 or representation == 2) and isinstance(filename, str)
def is_rgb(im: np.ndarray) -> bool: return im.ndim == RGBDIM
def read_image(filename: str, representation: int) -> np.ndarray:
    if not is_valid_args(filename, representation):
        raise Exception("Please provide valid filename and representation code")
    try: im = imread(filename)
    except OSError: raise Exception("Filename should be valid image filename")
    if is_rgb(im) and (representation == GREYSCALE): return rgb2gray(im).astype(np.float32)
    elif not is_rgb(im) and (representation == COLOR): raise Exception("Converting greyscale to RGB is not supported")
    return im.astype(np.float32) / MAX_PIX_VAL
# # # # # # # # # # # # # # # # # # # # END OF COPY&PASTE FROM EX1 # # # # # # # # # # # # # # # # # # # #


def magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    get magnitude of image as float32 dtype
    """
    return np.sqrt(np.abs(x)**2 + np.abs(y)**2).astype(np.float32)


def dft_matrix(n: int, conj_factor: int) -> np.ndarray:
    """
    get a dft matrix based on vandermonde matrix of size nXn
    :param n: the dimensions of the matrix
    :param conj_factor: 1 if its the regular dft matrix, -1 if we want the conjugate matrix
    :return: vandermonde matrix of the form w = e^((-2*pi*i)/N) (or e^(2*pi*i)/N) if conjugate needed)
    """
    fx = np.arange(n)
    fx_vandermonde = fx * fx.reshape(n, 1)  # generate vandermonde matrix by outer-product of the signal
    dft_matrix = np.exp((conj_factor*-2j * np.pi * fx_vandermonde) / n)  # create DFT matrix from vandermonde
    return dft_matrix


def DFT(signal: np.ndarray) -> np.ndarray:
    """
    Calc the fourier transform of a given 1D signal, using the formula F(u)=W*f(x)
    for W - DFT matrix (vandermonde matrix of the form w = (-2*pi*i)/N), and f(x) the signal.
    :param signal: f(x) the signal in space domain
    :return: F(u) - the signal in the frequency domain
    """
    return dft_matrix(signal.shape[0], STANDARD).dot(signal)


def IDFT(fourier_signal: np.ndarray) -> np.ndarray:
    """
    Calc the inverse fourier transform of a given 1D fourier signal, using the formula f(x)=W^(-1)*F(u)/N
    for W^(-1) - the inverse DFT matrix of the form w = (2*pi*i)/N, and F(u) the fourier signal.
    :param fourier_signal: F(u) - the signal in the frequency domain
    :return: f(x) the signal in space domain
    """
    return dft_matrix(fourier_signal.shape[0], CONJUGATE).dot(fourier_signal) / fourier_signal.shape[0]


def DFT2(image: np.ndarray) -> np.ndarray:
    """
    Calc the fourier transform of a given 2D signal, using the formula F(u,v) = W_m * f(x,y) * W_n
    for W_n - DFT matrix (vandermonde matrix of the form w = (-2*pi*i)/n), and f(x,y) the signal.
    :param image: f(x,y) the 2D signal in space domain
    :return: F(u,v) - the 2D signal in the frequency domain
    """
    n = image.shape[1]
    return DFT(image).dot(dft_matrix(n, STANDARD))


def IDFT2(fourier_image: np.ndarray) -> np.ndarray:
    """
    Calc the inverse fourier transform of a given 2D fourier signal,
    using the formula f(x,y)=(W_m*)*F(u,v)*(W_n*)/M*N, for W_m* - the inverse DFT matrix of the
    form w = (2*pi*i)/m, and F(u,v) the fourier signal.
    :param fourier_image: F(u,v) - the 2D signal in the frequency domain
    :return: f(x,y) the 2D signal in space domain
    """
    n = fourier_image.shape[1]
    return IDFT(fourier_image).dot(dft_matrix(n, CONJUGATE)) / n


def conv_der(im: np.ndarray) -> np.ndarray:
    """
    compute the magnitude of image derivatives, using simple convolution with [1 0 -1].
    :param im: greyscale image of type float32.
    :return: the magnitude of image derivatives as greyscale image of type float32.
    """
    dev_vec = np.array([1, 0, -1]).reshape(3, 1).astype(np.float32)
    dx = sp_signal.convolve2d(im, dev_vec, 'same')
    dy = sp_signal.convolve2d(im, dev_vec.T, 'same')
    return magnitude(dx, dy)


def fourier_der(im: np.ndarray) -> np.ndarray:
    """
    compute the magnitude of image derivatives, using fourier transform.
    :param im: greyscale image of type float32.
    :return: the magnitude of image derivatives as greyscale image of type float32.
    """
    fourier_im = np.fft.fftshift(DFT2(im))
    m, n = fourier_im.shape
    x_const, y_const = (2j*np.pi)/n, (2j*np.pi)/m
    u = np.arange(-n/2, n/2, 1)
    v = np.arange(-m/2, m/2, 1).reshape(m, 1)
    dx = x_const * np.fft.ifftshift(fourier_im * u)  # C*F(u,v)*u and shift back
    dy = y_const * np.fft.ifftshift(fourier_im * v)  # same with v
    inv_dx, inv_dy = IDFT2(dx), IDFT2(dy)
    return magnitude(inv_dx, inv_dy)


def gaussian_kernel(size: int) -> np.ndarray:
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

    g = gaussian_kernel(kernel_size)
    blur_im = sp_signal.convolve2d(im, g, 'same', 'wrap')  # using boundaries='wrap' so we will get periodic
    return blur_im


def blur_fourier(im: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    perform image blurring with gaussian kernel in Fourier space.
    :param im: the input image to be blurred (greyscale float32 image).
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer)
    :return: blurry image (greyscale float32 image).
    """
    if min(im.shape) < kernel_size:  # if kernel_size bigger than min(dimensions) of the image
        raise Exception("kernel_size must be smaller or equal to the smallest dimension of the image")

    # create padded kernel with equal dimensions to the given image:
    m, n = im.shape
    g = np.zeros((m, n), np.float32)
    kernel = gaussian_kernel(kernel_size)
    min_row_idx = int(np.floor(m/2) - np.floor(kernel_size/2))
    max_row_idx = int(np.floor(m/2) + np.floor(kernel_size/2)) + 1
    min_col_idx = int(np.floor(n/2) - np.floor(kernel_size/2))
    max_col_idx = int(np.floor(n / 2) + np.floor(kernel_size / 2)) + 1
    g[min_row_idx:max_row_idx, min_col_idx:max_col_idx] = kernel  # put the kernel in center of g
    g = np.fft.ifftshift(g)  # shift center of g to (0,0)

    # apply the filter and take only real part:
    blur_im = np.real(IDFT2(DFT2(im) * (DFT2(g)))).astype(np.float32)
    return blur_im






