import os
import tensorflow.compat.v2 as tf

IMG_SIZE = 299
LABELS = ['cat', 'dog', 'neg']


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def decode_jpeg(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def normalize_image(image):
    image = (image / 127.5) - 1
    return image


def resize_image(image, img_size):
    image = tf.image.resize(image, (img_size, img_size))
    return image


def normalize_resize_image(image, img_size):
    return resize_image(normalize_image(image), img_size)


# def write_to_tensorboard(model, path):
#     @tf.function
#     def traceme(x):
#         return model(x)
#
#     writer = tf.summary.create_file_writer(path)
#     tf.summary.trace_on(graph=True, profiler=True)
#     expected_shape = model.input.get_shape().as_list()[1:]
#     traceme(tf.zeros([1] + expected_shape))
#     with writer.as_default():
#         tf.summary.trace_export(
#             name="classifier_graph",
#             step=0,
#             profiler_outdir=path)


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=normalize_image)