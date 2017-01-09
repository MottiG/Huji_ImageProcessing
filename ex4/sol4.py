from sol4_utils import *
from sol4_add import *
from scipy.ndimage import interpolation

DER_FILTER = np.array([1, 0, -1], np.float32).reshape(1, 3)
BLUR_SIZE = 3
K = 0.04
N = M = 4  # defaults for spread_out_corners n and m


def harris_corner_detector(im: np.ndarray) -> np.ndarray:
    """
    extract harris-corner key feature points
    :param im: the image to extract key points
    :return: An array with shape (N,2) of [x,y] key points locations in im.
    """
    ix = sp_signal.convolve2d(im, DER_FILTER, 'same')
    iy = sp_signal.convolve2d(im, DER_FILTER.T, 'same')
    ix_2 = blur_spatial(ix**2, BLUR_SIZE)
    iy_2 = blur_spatial(iy**2, BLUR_SIZE)
    ix_iy = blur_spatial(ix * iy, BLUR_SIZE)
    r = ix_2 * iy_2 - ix_iy**2 - K*(ix_2 + iy_2)**2  # R - the response of the
    max_of_r = non_maximum_suppression(r)
    pos = np.transpose(np.nonzero(max_of_r))  # array of the indices of non-zero pixels in max_of_r
    pos_for_spread = np.transpose(np.array([pos[:, 1], pos[:, 0]]))  # school function works with [yx]
    return pos_for_spread


def sample_descriptor(im: np.ndarray, pos: np.ndarray, desc_rad: int) -> np.ndarray:
    n = pos.shape[0]
    desc = np.zeros((1+2*desc_rad, 1+2*desc_rad, n), np.float32)
    # for i in range(n):


