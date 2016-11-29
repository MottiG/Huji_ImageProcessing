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
    Calc the fourier transform of a given signal, using the formula F(u)=W*f(x)
    for W - DFT matrix (vandermonde matrix of the form w = (-2*pi*i)/N), and f(x) the signal.
    :param signal: f(x) the signal in his standard domain
    :return: F(u) - the signal in the frequency domain
    """
    n = signal.shape[0]
    return dft_matrix(n, STANDARD).dot(signal)


def IDFT(fourier_signal: np.ndarray) -> np.ndarray:
    """
    Calc the inverse fourier transform of a given fourier signal, using the formula f(x)=W^(-1)*F(u)/N
    for W^(-1) - the inverse DFT matrix of the form w = (2*pi*i)/N, and F(u) the fourier signal.
    :param fourier_signal: F(u) - the signal in the frequency domain
    :return: f(x) the signal in his standard domain
    """
    n = fourier_signal.shape[0]
    return dft_matrix(n, CONJUGATE).dot(fourier_signal) / n


# TODO - the DFT and IDFT return bad results for matrix inputs (not the same as np.fft.fft) - check it.
