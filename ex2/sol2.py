import numpy as np
import matplotlib.pyplot as plt

STANDARD = 1
CONJUGATE = -1


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
    Calc the inverse fourier transform of a given 2D fourier signal, using the formula f(x,y)=(W_m*)*F(u,v)*(W_n*)/M*N
    for W_m* - the inverse DFT matrix of the form w = (2*pi*i)/m, and F(u,v) the fourier signal.
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
    dev_vec = [1, 0, -1]

