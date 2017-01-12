from sol4_utils import *
from sol4_add import *
from scipy.ndimage import interpolation

DER_FILTER = np.array([1, 0, -1], np.float32).reshape(1, 3)
BLUR_SIZE = 3
K = 0.04
N = M = 4  # defaults for spread_out_corners n and m
RADIUS = 3
DEFAULT_DESC_RAD = 3
DEFAULT_MIN_SCORE = 0.5

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
    """
    sample a given image with simplified version of the MOPS descriptor
    :param im: grayscale image to sample within.
    :param pos: An array with shape (N,2) of [x,y] positions to sample descriptors in im
    :param desc_rad: "Radius" of descriptors to compute
    :return: A 3D array with shape (K,K,N) containing the ith descriptor at desc[:,:,i]
    """
    n = pos.shape[0]
    desc = np.empty((1 + 2*desc_rad, 1 + 2*desc_rad, n), np.float32)
    for i in range(n):
        grid = np.mgrid[pos[i, 0] - desc_rad: pos[i, 0] + desc_rad + 1,
                        pos[i, 1] - desc_rad: pos[i, 1] + desc_rad + 1]  # get the grid of indices
        desc_win = interpolation.map_coordinates(im, grid, order=1, prefilter=False)  # get the actual window
        win_mean = np.mean(desc_win)
        win_std = np.linalg.norm(desc_win - win_mean)
        if win_std:
            desc_win = (desc_win - win_mean) / win_std
        else:  # if the win is const, then the std will be zero
            desc_win[:] = 0
        desc[:, :, i] = desc_win

    return desc


def find_features(pyr: list) -> tuple:
    """
    get simplified version of the MOPS descriptors from the given pyramid and the positions of them
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels
    :return: the positions of the descriptors and array of them [with shape (1+2*desc_rad,1+2*desc_rad,N)]
    """
    pos = spread_out_corners(pyr[0], N, M, RADIUS)
    desc = sample_descriptor(pyr[2], pos/4, DEFAULT_DESC_RAD)
    return pos, desc


def match_features(desc1: np.ndarray, desc2: np.ndarray, min_score: float):
    """
    get the indices of matching descriptors between tow arrays of descriptors (desc1 and desc2)
    :param desc1: A feature descriptor array with shape(1+2*desc_rad,1+2*desc_rad,N1).
    :param desc2: A feature descriptor array with shape(1+2*desc_rad,1+2*desc_rad,N2).
    :param min_score: min score between two descriptors required to be regarded as corresponding points.
    :return: 2 arrays with shape (M,) and dtype int, of matching indices in the given descs (each for one)
    """
    # reshape so we can do dot product
    desc1 = desc1.reshape(desc1.shape[0]**2, desc1.shape[2])
    desc2 = desc2.reshape(desc2.shape[0]**2, desc2.shape[2])
    sjk = desc1.T.dot(desc2)
    sjk[sjk < min_score] = 0  # apply nin_score condition

    idx_of_desc1_max = np.argpartition(sjk, -2, 1)[:, -2:]
    idx_of_desc2_max = np.argpartition(sjk, -2, 0)[-2:, :].T

    desc1_ind = np.zeros((desc1.shape[1], desc2.shape[1]))
    for i in range(idx_of_desc1_max.shape[0]):
        desc1_ind[i, idx_of_desc1_max[i, :]] = 1

    desc2_ind = np.zeros((desc2.shape[1], desc1.shape[1]))
    for j in range(idx_of_desc2_max.shape[0]):
        desc2_ind[j, idx_of_desc2_max[j, :]] = 1

    # only if both desc1 and desc2 agree on descriptor, it will remain 1
    match_ind1, match_ind2 = np.where(desc1_ind * desc2_ind.T == 1)

    return match_ind1, match_ind2













