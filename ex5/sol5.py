from keras.layers import Input , merge, Convolution2D, Activation
from keras.models import Model
from keras.optimizers import adam
import numpy as np

NORM_VAL = 0.5  # value to subtract from each image
KERNEL_SIZE = 3
NUM_OF_BLOCKS = 5  # default number of rsblocks of the network
LOSS_FUNC = 'mean_square_error'
PERCENT_OF_TRAIN = 0.8
BETA_2 = 0.9  # for adam optimizer


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
    :return: symbolic output tensor of the layer conguration
    """
    layer = Convolution2D(num_channels , KERNEL_SIZE, KERNEL_SIZE, border_mode ='same')(input_tensor)
    layer = Activation('relu')(layer)
    layer = Convolution2D(num_channels, KERNEL_SIZE, KERNEL_SIZE, border_mode='same')(layer)
    output_tensor = merge([input_tensor, layer], mode ='sum')
    return output_tensor


def build_nn_model(height: int, width: int, num_channels: int) -> Model:
    """

    :param height: height of tensor output of the model
    :param width: width of tensor output of the model
    :param num_channels: number of channels for each of the convolutional layers except the last one
    :return: the network model
    """
    inpt = Input(shape=(GREYSCALE, height, width))
    layer = Convolution2D(num_channels, KERNEL_SIZE, KERNEL_SIZE, border_mode='same')(inpt)
    layer = Activation('relu')(layer)
    block_inpt = layer
    for i in range(NUM_OF_BLOCKS):
        block_inpt = resblock(block_inpt, num_channels=num_channels)  # TODO check if needed another variable as "out"
    layer = merge([layer, block_inpt], mode ='sum')
    last_conv = Convolution2D(1, KERNEL_SIZE, KERNEL_SIZE, border_mode='same')(layer)
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
    train_set = load_dataset(images[:train_size], batch_size, corruption_func, model.input_shape)
    valid_set = load_dataset(images[train_size:], batch_size, corruption_func, model.input_shape)
    model.compile(adam(beta_2=BETA_2), LOSS_FUNC)
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
    corrupted_image -= NORM_VAL # TODO check if needed here, if not, where?
    fix_im = model.predict(corrupted_image[np.newaxis,...])[0]
    fix_im += NORM_VAL  # TODO check if needed here, if not, where?
    return fix_im.clip(0, 1)


def add_gaussian_noise(image: np.ndarray, min_sigma: float, max_sigma: float) -> np.ndarray:
    """
    add to every pixel of the input image a zero-mean gaussian random variable with standard deviation equal to sigma
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param min_sigma:  a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma:  a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution.
    :return: corrupted image as np.ndarray with values in the [0, 1] range of type float32
    """
    sigma = (max_sigma - min_sigma)*np.random.random_sample() + min_sigma
    noise_im = np.random.normal(scale=sigma, size=image.shape)
    image += noise_im
    return image.clip(0, 1)




