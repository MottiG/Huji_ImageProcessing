from keras.layers import Input, merge, Convolution2D, Activation
from keras.models import Model
from keras.optimizers import adam
import numpy as np
from scipy.ndimage import filters
import sol5_utils
import random

# # # # # # # # # # # # #  START OF COPY&PASTE FROM EX1 - FOR READ_IMAGE FUNCTION # # # # # # # # # # # # # #
from scipy.misc import imread
from skimage.color import rgb2gray
GREYSCALE, COLOR, RGBDIM = 1, 2, 3
MAX_PIX_VAL = 255
def is_rgb(im: np.ndarray) -> bool:
    return im.ndim == RGBDIM
def read_image(filename: str, representation: int) -> np.ndarray:
    im = imread(filename)
    if is_rgb(im) and (representation == GREYSCALE): return rgb2gray(im).astype(np.float32)
    return im.astype(np.float32) / MAX_PIX_VAL
# # # # # # # # # # # # # # # # # # # # END OF COPY&PASTE FROM EX1 # # # # # # # # # # # # # # # # # # # # # #

NORM_VAL = 0.5  # value subtracted from each image to normalize mean
CONV_KERNEL_SIZE = 3  # default size of the kernels using as convolution layer
NUM_OF_BLOCKS = 5  # default number of resblocks of the network
LOSS_FUNC = 'mean_squared_error'
PERCENT_OF_TRAIN = 0.8
BETA_2 = 0.9  # for 'adam' optimizer

# denoising parameters:
DENOIS_PATCH_SIZE = 24
DENOIS_CHANNELS = 48
DENOIS_MIN_SIG = 0.0
DENOIS_MAX_SIG = 0.2
DENOIS_BATCH_SIZE = 100
DENOIS_TEST_BATCH_SIZE = 10
DENOIS_EPOCH_SIZE = 10000
DENOIS_TEST_EPOCH_SIZE = 30
DENOIS_EPOCH_NB = 5
DENOIS_TEST_EPOCH_NB = 2
DENOIS_VALID_SIZE = 1000
DENOIS_TEST_VALID_SIZE = 30

# deblurring parameters:
DEBLUR_PATCH_SIZE = 16
DEBLUR_CHANNELS = 32
DEBLUR_BATCH_SIZE = 100
DEBLUR_TEST_BATCH_SIZE = 10
DEBLUR_EPOCH_SIZE = 10000
DEBLUR_TEST_EPOCH_SIZE = 30
DEBLUR_EPOCH_NB = 10
DEBLUR_TEST_EPOCH_NB = 2
DEBLUR_VALID_SIZE = 1000
DEBLUR_TEST_VALID_SIZE = 30
DEBLUR_KERNEL_SIZE = 7


def load_dataset(filenames: list, batch_size: int, corruption_func: callable, crop_size: tuple):
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
    im_dict = {}

    while True:
        target_batch = np.empty((batch_size, GREYSCALE, crop_rows, crop_cols), np.float32)
        source_batch = np.empty((batch_size, GREYSCALE, crop_rows, crop_cols), np.float32)
        for i in range(batch_size):
            im_name = filenames[np.random.randint(max_num)]
            if im_name in im_dict:
                im = im_dict[im_name]
            else:
                im = read_image(im_name, GREYSCALE)
                im_dict[im_name] = im
            crop_im = corruption_func(im)
            # take patch
            max_patch_y, max_patch_x = im.shape[0] - crop_rows, im.shape[1] - crop_cols
            patch_y = np.random.randint(max_patch_y + 1)
            patch_x = np.random.randint(max_patch_x + 1)
            im_patch = im[patch_y: patch_y+crop_rows, patch_x: patch_x+crop_cols] - NORM_VAL
            crop_patch = crop_im[patch_y: patch_y+crop_rows, patch_x: patch_x+crop_cols] - NORM_VAL
            target_batch[i, ...] = im_patch[np.newaxis, ...]
            source_batch[i, ...] = crop_patch[np.newaxis, ...]

        yield source_batch, target_batch


def resblock(input_tensor: np.ndarray, num_channels: int) -> np.ndarray:
    """
    A residual block as described in ex. pdf
    :param input_tensor: a symbolic input tensor
    :param num_channels: number of channels for each of the convolutional layers
    :return: symbolic output tensor of the layer configuration
    """
    conv1 = Convolution2D(num_channels, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode ='same')(input_tensor)
    actv = Activation('relu')(conv1)
    conv2 = Convolution2D(num_channels, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(actv)
    output_tensor = merge([input_tensor, conv2], mode ='sum')
    return output_tensor


def build_nn_model(height: int, width: int, num_channels: int) -> Model:
    """

    :param height: height of tensor output of the model
    :param width: width of tensor output of the model
    :param num_channels: number of channels for each of the convolutional layers except the last one
    :return: the network model
    """
    inpt = Input(shape=(GREYSCALE, height, width))
    conv = Convolution2D(num_channels, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(inpt)
    actv = Activation('relu')(conv)
    block_inpt = actv
    for i in range(NUM_OF_BLOCKS):
        out = resblock(block_inpt, num_channels=num_channels)
        block_inpt = out
    layer = merge([actv, block_inpt], mode ='sum')
    last_conv = Convolution2D(1, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(layer)
    return Model(inpt, last_conv)


def train_model(model: Model, images: list, corruption_func: callable, batch_size: int,
                samples_per_epoch: int, num_epochs: int, num_valid_samples: int):
    """
    train the model to recognize corrupted images
    :param model: a general neural network model for image restoration.
    :param images: a list of files paths pointing to image files.
    :param corruption_func: A function receiving a numpy's array representation of an image as a single
    argument, and returns a randomly corrupted version of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch (actual samples, not batches!).
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    :return:
    """
    train_size = int(PERCENT_OF_TRAIN*len(images))
    crop_size = (model.input_shape[2], model.input_shape[3])
    train_set = load_dataset(images[:train_size], batch_size, corruption_func, crop_size)
    valid_set = load_dataset(images[train_size:], batch_size, corruption_func, crop_size)
    model.compile(loss=LOSS_FUNC, optimizer=adam(beta_2=BETA_2))
    model.fit_generator(generator=train_set, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=valid_set, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image: np.ndarray, base_model: Model, num_channels: int) -> np.ndarray:
    """
    restore a given  image
    :param corrupted_image: grayscale image of shape (height, width) and with values in the [0, 1]
    :param base_model: a neural network trained to restore small patches
    :param num_channels: number of channels used in the base model
    :return: image with values in the [−0.5, 0.5] range
    """
    model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels)
    model.set_weights(base_model.get_weights())
    fix_im = corrupted_image - NORM_VAL
    fix_im = model.predict(fix_im[np.newaxis, np.newaxis, ...])[0][0]
    fix_im += NORM_VAL
    return fix_im.clip(0, 1)


def add_gaussian_noise(image: np.ndarray, min_sigma: float, max_sigma: float) -> np.ndarray:
    """
    add to every pixel of the input image a zero-mean gaussian random variable with standard deviation equal
    to sigma
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param min_sigma:  a non-negative scalar value representing the minimal variance of the gaussian
    distribution.
    :param max_sigma:  a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution.
    :return: corrupted image as np.ndarray with values in the [0, 1] range of type float32
    """
    sigma = random.uniform(min_sigma, max_sigma)
    noise_im = np.random.normal(scale=sigma, size=image.shape)
    corrupted = image + noise_im
    return corrupted.clip(0, 1)


def learn_denoising_model(quick_mode: bool=False) -> tuple:
    """
    train a network to fix noising images
    :param quick_mode: if true, train with test-parameters to save time
    :return: tuple contains a trained denoising model, and the number of channels used in its construction
    """
    im_list = sol5_utils.images_for_denoising()
    num_channels = DENOIS_CHANNELS
    model = build_nn_model(height=DENOIS_PATCH_SIZE, width=DENOIS_PATCH_SIZE, num_channels=num_channels)
    train_model(model = model,
                images = im_list,
                corruption_func = lambda x: add_gaussian_noise(x, DENOIS_MIN_SIG, DENOIS_MAX_SIG),
                batch_size = DENOIS_BATCH_SIZE if not quick_mode else DENOIS_TEST_BATCH_SIZE,
                samples_per_epoch = DENOIS_EPOCH_SIZE if not quick_mode else DENOIS_TEST_EPOCH_SIZE,
                num_epochs = DENOIS_EPOCH_NB if not quick_mode else DENOIS_TEST_EPOCH_NB,
                num_valid_samples = DENOIS_VALID_SIZE if not quick_mode else DENOIS_TEST_VALID_SIZE)
    return model, num_channels


def add_motion_blur(image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
    """
    simulate motion blurring on a given image
    :param image: a grayscale image with values in the [0, 1] range
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: an angle in radians in the range [0, PI)
    :return: motion blurred image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    blurred = filters.convolve(image, kernel, mode='mirror')
    return blurred


def random_motion_blur(image: np.ndarray, list_of_kernel_sizes: list) -> np.ndarray:
    """
    simulate random motion blurring on a given image
    :param image: a grayscale image with values in the [0, 1] range
    :param list_of_kernel_sizes: a list of odd integers
    :return: randomly motion blurred image
    """
    angel = random.uniform(0, np.pi)
    kernel_size = list_of_kernel_sizes[np.random.randint(len(list_of_kernel_sizes))]
    return add_motion_blur(image, kernel_size, angel)


def learn_deblurring_model(quick_mode: bool=False) -> tuple:
    """
    train a network to fix blurring images
    :param quick_mode: if true, train with test-parameters to save time
    :return: tuple contains a trained deblurring model, and the number of channels used in its construction
    """
    im_list = sol5_utils.images_for_deblurring()
    num_channels = DEBLUR_CHANNELS
    model = build_nn_model(height=DEBLUR_PATCH_SIZE, width=DEBLUR_PATCH_SIZE, num_channels=num_channels)
    train_model(model = model,
                images = im_list,
                corruption_func = lambda x: random_motion_blur(x, [DEBLUR_KERNEL_SIZE]),
                batch_size = DEBLUR_BATCH_SIZE if not quick_mode else DEBLUR_TEST_BATCH_SIZE,
                samples_per_epoch = DEBLUR_EPOCH_SIZE if not quick_mode else DEBLUR_TEST_EPOCH_SIZE,
                num_epochs = DEBLUR_EPOCH_NB if not quick_mode else DEBLUR_TEST_EPOCH_NB,
                num_valid_samples = DEBLUR_VALID_SIZE if not quick_mode else DEBLUR_TEST_VALID_SIZE)
    return model, num_channels
