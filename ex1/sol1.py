import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray


def read_image(filename: str, representation: int) -> np.array:
    im = imread(filename)
    im_float = im.astype(np.float32)
    #TODO check if orginal image is greyscale or rgb. convert to grey if representation==1
    return None

