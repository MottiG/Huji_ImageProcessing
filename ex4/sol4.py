from sol4_utils import *
from sol4_add import *
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation

DER_FILTER = np.array([1, 0, -1], np.float32).reshape(1, 3)
BLUR_SIZE = 3
K = 0.04
N = M = 4  # defaults for spread_out_corners n and m
RADIUS = 3
DEFAULT_DESC_RAD = 3
DEFAULT_MIN_SCORE = 0.5
NUM_OF_POINTS = 4


def harris_corner_detector(im: np.ndarray) -> np.ndarray:
    """
    extract harris-corner key feature points
    :param im: the image to extract key points
    :return: An array with shape (N,2) of [x,y] key points locations in im.
    """
    ix = sp_signal.convolve2d(im, DER_FILTER, 'same', 'wrap')
    iy = sp_signal.convolve2d(im, DER_FILTER.T, 'same', 'wrap')
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
    k = 1 + 2*desc_rad
    n = pos.shape[0]
    desc = np.empty((k, k, n), np.float32)
    for i in range(n):
        grid = np.meshgrid(np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1),
                           np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1))
        desc_win = interpolation.map_coordinates(im, grid, order=1, prefilter=False)
        desc_win -= np.mean(desc_win)
        win_std = np.linalg.norm(desc_win)
        if win_std:
            desc_win /= win_std
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


def match_features(desc1: np.ndarray, desc2: np.ndarray, min_score: float) -> tuple:
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


def apply_homography(pos1: np.ndarray, H12: np.ndarray) -> np.ndarray:
    """
    apply an homography transformation on a set of points
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates in image i+1
    obtained from transforming pos1 using H12.
    """
    pos1 = np.hstack((pos1, np.ones((pos1.shape[0], 1))))  # add homographic element
    trans = pos1.dot(H12.T)
    pos2 = trans[:, [0, 1]] / trans[:, 2].reshape(trans.shape[0], 1)  # normalize back to x,y
    return pos2


def ransac_homography(pos1: np.ndarray, pos2: np.ndarray, num_iters: int, inlier_tol: np.float32) -> tuple:
    """
    apply RANSAC homography fitting
    :param pos1: an array with n rows of [x,y] coordinates of matched points of first image.
    :param pos2: an array with n rows of [x,y] coordinates of matched points of second image.
    :param num_iters: number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :return: A 3x3 normalized homography matrix and An Array with shape (S,) where S is the number
    of inliers, containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    inliers = np.array([])

    for i in range(num_iters):
        rand_idx = np.random.choice(pos1.shape[0], size=NUM_OF_POINTS)
        pos1_smpl, pos2_smpl = pos1[rand_idx, :], pos2[rand_idx, :]
        h = least_squares_homography(pos1_smpl, pos2_smpl)
        if h is None:
            continue
        pos2_trans = apply_homography(pos1, h)
        e = np.linalg.norm(pos2_trans - pos2, axis=1)**2
        curr_inliers = np.where(e < inlier_tol)[0]  # indices of "good" points of pos2
        if len(curr_inliers) > len(inliers):
            inliers = curr_inliers

    H12 = least_squares_homography(pos1[inliers, :], pos2[inliers, :])

    return H12, inliers


def display_matches(im1, im2, pos1, pos2, inliers) -> None:
    """
    visualize the full set of point matches and the inlier matches detected by RANSAC
    :param im1: first grayscale image
    :param im2: second grayscale images
    :param pos1: an array with n rows of [x,y] coordinates of matched points of first image.
    :param pos2: an array with n rows of [x,y] coordinates of matched points of second image.
    :param inliers: An array with shape (S,) of inlier matches (e.g. see output of ransac homography)
    """
    im = np.hstack((im1, im2))
    pos2[:, 0] += im1.shape[1]
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    for i in range(len(pos1)):
        color = 'y' if i in inliers else 'b'
        plt.plot([pos1[i, 0], pos2[i, 0]], [pos1[i, 1], pos2[i, 1]], mfc='r', c=color, lw=1, ms=5, marker='.')
    plt.show()



