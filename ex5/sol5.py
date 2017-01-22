from keras.layers import Input , Dense, Convolution2D, Activation
from keras.models import Model
import numpy as np

SUBVAL = 0.5  # value to subtract from each image
KERNEL_SIZE = 3

# # # # # # # # # # START OF COPY&PASTE FROM EX1 - THE PRESUBMIT USES THE READ_IMAGE FUNCTION # # # # # # # #
from scipy.misc import imread
from skimage.color import rgb2gray
GREYSCALE, COLOR, RGBDIM = 1, 2, 3
MAX_PIX_VAL = 255
def is_valid_args(filename: str, representation: int) -> bool:
    return (filename is not None) and (representation == 1 or representation == 2) and isinstance(filename, str)
def is_rgb(im: np.ndarray) -> bool: return im.ndim == RGBDIM
def read_image(filename: str, representation: int) -> np.ndarray:
    if not is_valid_args(filename, representation): raise Exception("Please provide valid filename and representation code")
    try: im = imread(filename)
    except OSError: raise Exception("Filename should be valid image filename")
    if is_rgb(im) and (representation == GREYSCALE): return rgb2gray(im).astype(np.float32)
    elif not is_rgb(im) and (representation == COLOR): raise Exception("Converting greyscale to RGB is not supported")
    return im.astype(np.float32) / MAX_PIX_VAL
# # # # # # # # # # # # # # # # # # # # END OF COPY&PASTE FROM EX1 # # # # # # # # # # # # # # # # # # # # # #


def load_dataset(filenames: list, batch_size: int, corruption_func: callable, crop_size: tuple) -> tuple:
    """
    get generator to generate random data, see ex. pdf for more details.
    :param filenames: A list of file names of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy's array representation of an image as a single
    argument, and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: Python's generator object which outputs random tuples of the form (source_batch, target_batch)
    """
    max_num = len(filenames)
    crop_rows, crop_cols = crop_size
    im_dict = {}  # TODO check if should be here or inside the while

    while True:

        target_batch = source_batch = np.empty((batch_size, GREYSCALE, crop_rows, crop_cols), np.float32)
        for i in range(batch_size):
            im_name = filenames[np.random.randint(max_num)]
            if im_name in im_dict:
                im = im_dict[im_name]
            else:
                im = read_image(im_name, GREYSCALE)
                im_dict[im_name] = im
            # crop_im = corruption_func(im)
            # take patch
            max_patch_y, max_patch_x = im.shape[0] - crop_rows, im.shape[1] - crop_cols
            patch_y = np.random.randint(max_patch_y + 1)
            patch_x = np.random.randint(max_patch_x + 1)
            im_patch = im[patch_y: patch_y+crop_rows, patch_x: patch_x+crop_cols] - SUBVAL
            # crop_patch = crop_im[patch_y: patch_y+crop_rows, patch_x: patch_x+crop_cols] - SUBVAL
            target_batch[i, ...] = im_patch[np.newaxis, ...]
            # source_batch[i, ...] = crop_patch[np.newaxis, ...]

        yield source_batch, target_batch


def resblock(input_tensor: np.ndarray, num_channels: int) -> np.ndarray:
    """
    A residual block as described in ex. pdf
    :param input_tensor: a symbolic input tensor
    :param num_channels: number of channels for each of the convolutional layers
    :return: symbolic output tensor of the layer conguration
    """
    # inpt = Input(shape=input_tensor.shape)  # TODO check if needed or send input directly to conv
    first_conv =  Convolution2D(num_channels , KERNEL_SIZE, KERNEL_SIZE, border_mode ='same ')(input_tensor)
    actv = Activation('relu')(first_conv)
    second_conv = Convolution2D(num_channels, KERNEL_SIZE, KERNEL_SIZE, border_mode='same ')(actv)
    output_tensor = input_tensor + second_conv
    return output_tensor


