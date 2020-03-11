import numpy as np
import argparse
import tensorflow.compat.v2 as tf
from utils import IMG_SIZE, LABELS, image_generator
import tensorflow as tf1
tf1.compat.v1.enable_eager_execution()


def classify(model, test_dir):
    """
    Classifies all images in test_dir
    :param model: Model to be evaluated
    :param test_dir: Directory including the images
    :return: None
    """
    test_img_gen = image_generator.flow_from_directory(test_dir,
                                                       target_size=(IMG_SIZE, IMG_SIZE),
                                                       classes=LABELS,
                                                       batch_size=1,
                                                       shuffle=False)

    ######### Your code starts here #########
    # Classify all images in the given folder
    # Calculate the accuracy and the number of test samples in the folder
    # test_img_gen has a list attribute filenames where you can access the filename of the datapoint
    num_test = test_img_gen.samples
    correct_count = 0
    for i in range(num_test):
        x_i, y_i = next(test_img_gen)
        y_pred = model(x_i).numpy()
        if np.argmax(y_pred[0]) == np.argmax(y_i[0]):
            correct_count += 1
        else:
            print(test_img_gen.filenames[i])
    ######### Your code ends here #########
    accuracy = correct_count / num_test
    print(f"Evaluated on {num_test} samples.")
    print(f"Accuracy: {accuracy*100:.0f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_dir', type=str, default='datasets/test/')
    FLAGS, _ = parser.parse_known_args()
    model = tf.keras.models.load_model('./trained_models/trained.h5')
    classify(model, FLAGS.test_image_dir)