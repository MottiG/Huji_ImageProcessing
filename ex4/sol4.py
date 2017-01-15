from sol4_utils import *
from sol4_add import *
from itertools import accumulate
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates

DER_FILTER = np.array([1, 0, -1], np.float32).reshape(1, 3)
BLUR_SIZE = 3
K = 0.04
N = M = 4  # defaults for spread_out_corners n and m
RADIUS = 3
DEFAULT_DESC_RAD = 3
DEFAULT_MIN_SCORE = 0.5
NUM_OF_POINTS_TO_TRANS = 4
EPSILON = 10 ** -5
OVERLAP = 15


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
    k = 1+2*desc_rad
    n = pos.shape[0]
    desc = np.empty((k, k, n), np.float32)
    for i in range(n):
        grid = np.meshgrid(np.arange(pos[i, 1] - desc_rad, pos[i, 1] + desc_rad + 1),
                           np.arange(pos[i, 0] - desc_rad, pos[i, 0] + desc_rad + 1))
        desc_win = map_coordinates(im, grid, order=1, prefilter=False)
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


def get_binary_mat(idx_of_max: np.ndarray, shape: tuple) -> np.ndarray:
    """
    get binary matrix with once where the indices of descriptors are with max value
    :param idx_of_max: matrix with the indices of max values
    :param shape: the shape of the desire matrix
    :return: a binary matrix with once where the indices of descriptors are with max value
    """
    binary_mat = np.zeros(shape)
    for i in range(idx_of_max.shape[0]):
        binary_mat[i, idx_of_max[i, :]] = 1
    return binary_mat


def match_features(desc1: np.ndarray, desc2: np.ndarray, min_score: float) -> tuple:
    """
    get the indices of matching descriptors between tow arrays of descriptors (desc1 and desc2)
    :param desc1: A feature descriptor array with shape(1+2*desc_rad,1+2*desc_rad,N1).
    :param desc2: A feature descriptor array with shape(1+2*desc_rad,1+2*desc_rad,N2).
    :param min_score: min score between two descriptors required to be regarded as corresponding points.
    :return: 2 arrays with shape (M,) and dtype int, of matching indices in the given descs (each for one)
    """
    # reshape so every column is a single flatten descriptor
    desc1 = desc1.reshape(desc1.shape[0]**2, desc1.shape[2])
    desc2 = desc2.reshape(desc2.shape[0]**2, desc2.shape[2])
    sjk = desc1.T.dot(desc2)
    sjk[sjk < min_score] = 0  # apply nin_score condition

    idx_of_desc1_max = np.argpartition(sjk, -2, 1)[:, -2:]
    idx_of_desc2_max = np.argpartition(sjk, -2, 0)[-2:, :].T

    # get binary matrices represent the indices of max descriptors
    desc1_ind = get_binary_mat(idx_of_desc1_max, (desc1.shape[1], desc2.shape[1]))
    desc2_ind = get_binary_mat(idx_of_desc2_max, (desc2.shape[1], desc1.shape[1]))

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
    trans = pos1.dot(H12.T).astype(np.float32)  # its is more efficient to transpose H12 and not pos1
    trans[:, 2][trans[:, 2] == 0.0] = EPSILON  # prevent divide by zero  # TODO check if ok
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
        rand_idx = np.random.choice(pos1.shape[0], size=NUM_OF_POINTS_TO_TRANS)  # choose 4 points
        pos1_smpl, pos2_smpl = pos1[rand_idx, :], pos2[rand_idx, :]
        h = least_squares_homography(pos1_smpl, pos2_smpl)
        if h is None:
            continue
        pos1_trans = apply_homography(pos1, h)
        e = np.linalg.norm(pos1_trans - pos2, axis=1)**2
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


def accumulate_homographies(H_successive: list, m: int) -> list:
    """
    get Hi,m from {Hi,i+1 : i = 0..M-1}.
    :param H_successive: A list of 3x3 homography matrices where H successive[i] is a homography
    that transforms points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system we would like to accumulate the given homographies towards.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i
    to coordinate system m.
    """
    less_than_m = H_successive[:m]  # all matrices for i<m
    less_than_m = list(accumulate(less_than_m[::-1], np.dot))[::-1]  # reverse again to 0-m
    less_than_m.append(np.eye(3))  # add for i=m
    bigger_than_m = H_successive[m:]  # all matrices for i>m
    bigger_than_m = list(map(np.linalg.inv, bigger_than_m))
    bigger_than_m = list(accumulate(bigger_than_m, np.dot))
    H2m = np.array(less_than_m + bigger_than_m)
    H2m = H2m.T / H2m[:, 2, 2]
    return list(H2m.T)


def next_power(d: int):
    """
    :param d: size of dimension of image
    :return: the next power of 2 of this size
    """
    res = 1
    while res < d: res <<= 1
    return res


def get_centers_and_corners(ims: list, Hs: list) -> tuple:
    """
    get the corners and centers of each im in ims
    :param ims: list of grayscale images.
    :param Hs: list of 3x3 homography matrices.
    :return: tuple containing a list of x,y centers  and  a list with x_min,x_max,y_min and y_max
    """
    centers = []
    corners = np.empty((len(ims), 4))
    for i in range(len(ims)):
        rows_cor, cols_cor = ims[i].shape[0] - 1, ims[i].shape[1] - 1

        # get center of im[i]:
        curr_center = np.array([cols_cor/2, rows_cor/2]).reshape(1, 2)
        centers.append(apply_homography(curr_center, Hs[i])[0])

        # get corners of im[i]:
        curr_cornrs = np.array([[0, 0], [cols_cor, 0], [0, rows_cor], [cols_cor, rows_cor]]).reshape(4, 2)
        curr_cornrs = apply_homography(curr_cornrs, Hs[i])
        corners[i, 0] = np.min(curr_cornrs[:, 0])  # curr_x_min
        corners[i, 1] = np.max(curr_cornrs[:, 0])  # curr_x_max
        corners[i, 2] = np.min(curr_cornrs[:, 1])  # curr_y_min
        corners[i, 3] = np.max(curr_cornrs[:, 1])  # curr_y_max

    # calc canvas corners
    x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 1])
    y_min, y_max = np.min(corners[:, 2]), np.max(corners[:, 3])
    corners = [x_min, x_max, y_min, y_max]
    return centers, corners


def render_panorama(ims: list, Hs: list) -> np.ndarray:
    """
    panorama rendering
    :param ims: list of grayscale images.
    :param Hs: list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the
    coordinate system of ims [i] to the coordinate system of the panorama.
    :return: A grayscale panorama image composed of vertical strips, backwarped using homographies from Hs,
    one from every image in ims.
    """
    if len(ims) == 1:
        return ims[0]

    # create the canvas of the panorama
    centers, corners = get_centers_and_corners(ims, Hs)  # corners = [x_min, x_max, y_min, y_max]
    x_pano, y_pano = np.meshgrid(np.arange(corners[0], corners[1]+1),
                                 np.arange(corners[2], corners[3]+1))
    panorama = np.zeros(x_pano.shape)  # the canvas of the panorama
    pan_rows, pan_cols = panorama.shape
    next_power_of_rows = next_power(pan_rows)  # will use later for padding
    next_power_of_cols = next_power(pan_cols)

    # create borders of strips
    borders = [int(np.round((centers[i][0] + centers[i+1][0])/2) - corners[0]) for i in range(len(ims)-1)]
    borders.insert(0, 0)
    borders.append(x_pano.shape[1])

    # apply panorama
    for i in range(len(ims)):
        left = borders[i] - OVERLAP if i != 0 else borders[i]
        right = borders[i+1] + OVERLAP if i != len(ims)-1 else borders[i+1]
        x_coord, y_coord = x_pano[:, left:right], y_pano[:, left:right]  # indices of the current part
        xi_yi = np.array([x_coord.flatten(), y_coord.flatten()]).T
        xi_yi = apply_homography(xi_yi, np.linalg.inv(Hs[i]))

        curr_im = map_coordinates(ims[i], [xi_yi[:, 1], xi_yi[:, 0]], order=1, prefilter=False)
        curr_im = curr_im.reshape(panorama[:, left:right].shape)
        # panorama[:, left:right] = curr_im  # TODO dell after implementing blending

        # apply blending on panorama:
        if i == 0:
            panorama[:, left:right] = curr_im
            continue
        temp_canvas = np.zeros(panorama.shape)
        temp_canvas[:, left:right] = curr_im

        # pad temp canvas:
        if next_power_of_rows > pan_rows:  # from top
            temp_canvas = np.vstack((np.zeros((next_power_of_rows-pan_rows, temp_canvas.shape[1])),
                                    temp_canvas))
            panorama = np.vstack((np.zeros((next_power_of_rows-pan_rows, panorama.shape[1])), panorama))
        if next_power_of_cols > pan_cols: # from right side
            temp_canvas = np.hstack((temp_canvas,
                                     np.zeros((temp_canvas.shape[0], next_power_of_cols-pan_cols))))
            panorama = np.hstack((panorama, np.zeros((panorama.shape[0], next_power_of_cols-pan_cols))))

        # create mask:
        mask = np.ones(panorama.shape)
        mask[:, borders[i]:borders[i+1]] = 0
        panorama = pyramid_blending(panorama, temp_canvas, mask, 3, 7, 7)

        # plt.imshow(panorama.clip(0, 1), cmap=plt.cm.gray) #TODO dell
        # plt.show() #TODO dell

        panorama = panorama[0:pan_rows, 0:pan_cols]

    return panorama

